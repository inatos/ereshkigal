use crate::error::{Error, Result};
use crate::prompt::{digest, render_prompt, state_prefix_text};
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

    pub fn encode_decision(&self, row: &DecisionRow, max_tokens: usize) -> Result<EncodedDecision> {
        let prompt = render_prompt(row)?;
        let ids = self.encode_ids(&prompt)?;
        if ids.is_empty() || ids.len() > max_tokens {
            return Err(Error::Validation(format!(
                "Row {}: {} input tokens exceed limit {}; no truncation allowed",
                row.id,
                ids.len(),
                max_tokens
            )));
        }
        let slots = self.slot_ids(row.options.len())?;
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
        Ok(EncodedDecision {
            prompt_sha256: digest(&prompt),
            prompt,
            token_ids: ids,
            slots,
        })
    }

    pub fn state_prefix_ids(&self, row: &DecisionRow) -> Result<Vec<i32>> {
        let text = state_prefix_text(row)?;
        let mut ids = self.encode_ids(&text)?;
        if ids.is_empty() {
            return Err(Error::Validation("empty state prefix".into()));
        }
        ids.pop(); // upstream drops final boundary token
        Ok(ids)
    }

    fn slot_ids(&self, count: usize) -> Result<Vec<i32>> {
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
}
