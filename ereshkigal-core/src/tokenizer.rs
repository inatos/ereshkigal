use crate::error::{Error, Result};
use crate::prompt::digest;
use crate::types::{DecisionRow, EncodedDecision, LETTERS};
use hf_hub::api::sync::ApiBuilder;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct ReferenceTokenizer {
    inner: Tokenizer,
    pub source: String,
    pub revision: String,
    pub path: PathBuf,
}

impl ReferenceTokenizer {
    pub fn from_file(path: impl AsRef<Path>, source: String, revision: String) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| Error::Tokenizer(format!("load {}: {e}", path.display())))?;
        Ok(Self {
            inner,
            source,
            revision,
            path,
        })
    }

    /// Download / reuse the pinned Hugging Face tokenizer.json.
    pub fn from_hub(source: &str, revision: &str) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            source.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        let path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Self::from_file(path, source.to_string(), revision.to_string())
    }

    pub fn encode_ids(&self, text: &str) -> Result<Vec<i32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    pub fn encode_ids_with_offsets(&self, text: &str) -> Result<(Vec<i32>, Vec<(usize, usize)>)> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let offsets = encoding.get_offsets().to_vec();
        Ok((ids, offsets))
    }

    pub fn encode_decision(&self, row: &DecisionRow, max_tokens: usize) -> Result<EncodedDecision> {
        self.encode_decision_version(row, max_tokens, crate::types::PROMPT_VERSION)
    }

    pub fn encode_decision_version(
        &self,
        row: &DecisionRow,
        max_tokens: usize,
        prompt_version: &str,
    ) -> Result<EncodedDecision> {
        let prompt = crate::prompt::render_prompt_version(row, prompt_version)?;
        let mut ids = self.encode_ids(&prompt)?;
        if ids.is_empty() {
            return Err(Error::Validation(format!(
                "Row {}: tokenizer produced no tokens",
                row.id
            )));
        }
        let truncated = ids.len() > max_tokens && max_tokens > 0;
        if truncated {
            ids = crate::engine::truncate_tokens_head_tail(&ids, max_tokens);
        }
        let slots = self.slot_ids(row.options.len())?;
        if !truncated {
            for (letter, &token) in LETTERS.chars().zip(slots.iter()) {
                let with_letter = self.encode_ids(&format!("{prompt}{letter}"))?;
                let mut expected = ids.clone();
                expected.push(token);
                if with_letter != expected {
                    return Err(Error::Validation(format!(
                        "Answer boundary changes tokenization for slot {letter}"
                    )));
                }
            }
        }
        Ok(EncodedDecision {
            prompt_sha256: digest(&prompt),
            prompt,
            token_ids: ids,
            slots,
        })
    }

    pub fn state_prefix_ids(&self, row: &DecisionRow) -> Result<Vec<i32>> {
        self.state_prefix_ids_version(row, crate::types::PROMPT_VERSION)
    }

    pub fn state_prefix_ids_version(&self, row: &DecisionRow, prompt_version: &str) -> Result<Vec<i32>> {
        let text = crate::prompt::state_prefix_text_version(row, prompt_version)?;
        let mut ids = self.encode_ids(&text)?;
        if ids.is_empty() {
            return Err(Error::Validation("empty state prefix".into()));
        }
        ids.pop(); // upstream drops final boundary token
        Ok(ids)
    }

    pub fn slot_ids(&self, count: usize) -> Result<Vec<i32>> {
        let mut result = Vec::with_capacity(count);
        for letter in LETTERS.chars().take(count) {
            let encoded = self.encode_ids(&letter.to_string())?;
            if encoded.len() != 1 {
                return Err(Error::Validation(format!(
                    "Answer slot {letter:?} is not one exact round-trip token"
                )));
            }
            // Decode check via re-encode stability of single char already done.
            result.push(encoded[0]);
        }
        let mut uniq = result.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() != result.len() {
            return Err(Error::Validation("Answer-slot tokens collide".into()));
        }
        Ok(result)
    }
}

/// Map strictly increasing prompt byte offsets (e.g. each FIELD's `ANSWER:` colon)
/// to their covering token indices in `offsets`. Returns `None` if any byte offset
/// cannot be uniquely mapped to a strictly increasing token index.
pub fn locate_token_positions(
    offsets: &[(usize, usize)],
    byte_offsets: &[usize],
) -> Option<Vec<usize>> {
    if byte_offsets.is_empty() {
        return None;
    }
    let mut positions = Vec::with_capacity(byte_offsets.len());
    let mut search_from = 0usize;
    for &byte_off in byte_offsets {
        let mut found = None;
        for (idx, &(start, end)) in offsets.iter().enumerate().skip(search_from) {
            if start < end && start <= byte_off && byte_off < end {
                found = Some(idx);
                break;
            }
        }
        let pos = found?;
        positions.push(pos);
        search_from = pos + 1;
    }
    Some(positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionSpec;
    use serde_json::json;
    use std::path::PathBuf;

    fn cached_tokenizer() -> Option<ReferenceTokenizer> {
        let home = std::env::var_os("HOME")?;
        let path = PathBuf::from(home)
            .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json");
        if !path.is_file() {
            return None;
        }
        ReferenceTokenizer::from_file(
            path,
            "Qwen/Qwen3.5-4B".into(),
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into(),
        )
        .ok()
    }

    #[test]
    fn encode_support1_matches_golden() {
        let Some(tok) = cached_tokenizer() else {
            eprintln!("skip: HF tokenizer cache missing");
            return;
        };
        let row = DecisionRow {
            id: "support-1".into(),
            state: json!(
                "The deployment completed at 14:02 UTC. Health checks passed in all three zones. No rollback was initiated."
            ),
            question: "Is there evidence that the deployment succeeded?".into(),
            options: vec![
                OptionSpec {
                    id: "yes".into(),
                    description: "The deployment succeeded.".into(),
                },
                OptionSpec {
                    id: "no".into(),
                    description: "The deployment did not succeed.".into(),
                },
                OptionSpec {
                    id: "insufficient".into(),
                    description: "The evidence is insufficient to decide.".into(),
                },
            ],
        };
        let enc = tok.encode_decision(&row, 4096).unwrap();
        assert_eq!(
            enc.prompt_sha256,
            "7ac35785358f0c656eeb741ca0a51f03e0133753a71e2879a18ad0ab56d0c024"
        );
        assert_eq!(enc.token_ids.len(), 142);
        assert_eq!(enc.slots, vec![32, 33, 34]);
        let prefix = tok.state_prefix_ids(&row).unwrap();
        assert_eq!(prefix.len(), 65);
        assert_eq!(&enc.token_ids[..prefix.len()], prefix.as_slice());
    }

    #[test]
    fn locate_token_positions_maps_strictly_increasing_spans() {
        let offsets = vec![(0, 6), (6, 12), (12, 13), (13, 20), (20, 22)];
        assert_eq!(locate_token_positions(&offsets, &[12, 20]), Some(vec![2, 4]));
        // Duplicate/out-of-order byte offsets in the same token fail cleanly.
        assert_eq!(locate_token_positions(&offsets, &[20, 21]), None);
        assert_eq!(locate_token_positions(&offsets, &[99]), None);
        assert_eq!(locate_token_positions(&offsets, &[]), None);
    }

    #[test]
    fn joint_prompt_colon_offsets_locate_in_qwen3_tokenizer() {
        let pack_tok = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../packs/qwen3-0.6b-lora-dev/tokenizer.json");
        let tok = if pack_tok.is_file() {
            ReferenceTokenizer::from_file(pack_tok, "Qwen/Qwen3-0.6B".into(), "local".into()).ok()
        } else {
            cached_tokenizer()
        };
        let Some(tok) = tok else {
            eprintln!("skip: no local Qwen3 tokenizer.json");
            return;
        };
        let state = json!("Server metrics: CPU 12%, latency 4ms, zero errors.");
        let rows = vec![
            DecisionRow {
                id: "q1".into(),
                state: state.clone(),
                question: "Is latency healthy?".into(),
                options: vec![
                    OptionSpec { id: "yes".into(), description: "Yes.".into() },
                    OptionSpec { id: "no".into(), description: "No.".into() },
                ],
            },
            DecisionRow {
                id: "q2".into(),
                state: state.clone(),
                question: "Are there errors?".into(),
                options: vec![
                    OptionSpec { id: "yes".into(), description: "Yes.".into() },
                    OptionSpec { id: "no".into(), description: "No.".into() },
                ],
            },
        ];
        let (prompt, colons) =
            crate::prompt::render_joint_batch_prompt_with_offsets(&state, &rows).unwrap();
        let (ids, offsets) = tok.encode_ids_with_offsets(&prompt).unwrap();
        let positions = locate_token_positions(&offsets, &colons).expect("colon token positions");
        assert_eq!(positions.len(), 2);
        assert!(positions[0] < positions[1]);
        assert!(positions[1] < ids.len());
    }
}
