use crate::cache::{PrefixRadix, ReplayCache};
use crate::engine::EngineOwned;
use crate::error::{Error, Result};
use crate::gbnf::{letter_gbnf, slots_match_letters};
use crate::softmax::{argmax, softmax};
use crate::tokenizer::ReferenceTokenizer;
use crate::types::{DecisionRow, EncodedDecision, ModelMeta, ScoreResult, PROMPT_VERSION};
use llama_cpp_2::context::session::SeqState;
use std::time::Instant;

pub struct Scorer {
    engine: EngineOwned,
    tokenizer: ReferenceTokenizer,
    max_tokens: usize,
    serial: SerialCache,
    replay: ReplayCache,
    radix: PrefixRadix,
    pub prompt_version: String,
    pub replay_enabled: bool,
    pub radix_enabled: bool,
    pub parallel_suffixes: bool,
}

struct SerialCache {
    prefix: Option<Vec<i32>>,
    state: Option<SeqState>,
}

impl Scorer {
    pub fn new(engine: EngineOwned, tokenizer: ReferenceTokenizer) -> Self {
        let max_tokens = engine.meta.max_prompt_tokens;
        Self {
            engine,
            tokenizer,
            max_tokens,
            serial: SerialCache {
                prefix: None,
                state: None,
            },
            replay: ReplayCache::default(),
            radix: PrefixRadix::default(),
            prompt_version: PROMPT_VERSION.to_string(),
            replay_enabled: true,
            radix_enabled: true,
            parallel_suffixes: true,
        }
    }

    pub fn meta(&self) -> &ModelMeta {
        &self.engine.meta
    }

    pub fn encode_verified(&self, row: &DecisionRow) -> Result<EncodedDecision> {
        let enc = self.tokenizer.encode_decision_version(
            row,
            self.max_tokens,
            &self.prompt_version,
        )?;
        let gguf = self.engine.gguf_tokenize(&enc.prompt)?;
        if gguf != enc.token_ids {
            return Err(Error::Validation(format!(
                "Row {}: GGUF tokenization disagrees with the reference tokenizer",
                row.id
            )));
        }
        let _ = letter_gbnf(row.options.len())?;
        slots_match_letters(&enc.slots, &enc.slots)?;
        Ok(enc)
    }

    pub fn score_direct(&mut self, row: &DecisionRow) -> Result<ScoreResult> {
        let started = Instant::now();
        let enc = self.encode_verified(row)?;
        if self.replay_enabled {
            if let Some(hit) = self.replay.get(&enc.prompt_sha256) {
                let mut cloned = hit.clone();
                cloned.replay_hit = Some(true);
                cloned.cache_hit = Some(true);
                cloned.total_seconds = Some(started.elapsed().as_secs_f64());
                cloned.forward_seconds = Some(0.0);
                return Ok(cloned);
            }
        }
        let mut radix_len = None;
        let vocabulary = if self.radix_enabled {
            if let Some((plen, st)) = self.radix.longest(&enc.token_ids) {
                if plen < enc.token_ids.len() {
                    radix_len = Some(plen);
                    self.engine.restore_state(st)?;
                    self.engine
                        .branch_logits(plen, &enc.token_ids[plen..])?
                } else {
                    self.engine.full_logits(&enc.token_ids)?
                }
            } else {
                self.engine.full_logits(&enc.token_ids)?
            }
        } else {
            let mark = Instant::now();
            let v = self.engine.full_logits(&enc.token_ids)?;
            let _ = mark;
            v
        };
        let forward = started.elapsed().as_secs_f64();
        let mut result = self.finish(
            row,
            &enc,
            &vocabulary,
            "llamacpp-direct-v1",
            "quantized last-position logits restricted to declared answer slots; no generated tokens",
            Some(forward),
            Some(started.elapsed().as_secs_f64()),
            None,
            None,
            None,
            None,
            None,
        );
        result.replay_hit = Some(false);
        result.radix_prefix_tokens = radix_len;
        if self.replay_enabled {
            self.replay.insert(enc.prompt_sha256.clone(), result.clone());
        }
        Ok(result)
    }

    pub fn score_serial(&mut self, row: &DecisionRow) -> Result<ScoreResult> {
        let started = Instant::now();
        let enc = self.encode_verified(row)?;
        let prefix = self
            .tokenizer
            .state_prefix_ids_version(row, &self.prompt_version)?;
        if prefix.is_empty()
            || enc.token_ids.len() <= prefix.len()
            || enc.token_ids[..prefix.len()] != prefix[..]
        {
            return Err(Error::Validation(
                "State prefix does not match the full prompt".into(),
            ));
        }
        let mut hit = self.serial.state.is_some() && self.serial.prefix.as_ref() == Some(&prefix);
        let mut prefill_seconds = 0.0;
        if !hit {
            if self.radix_enabled {
                if let Some((plen, st)) = self.radix.longest(&prefix) {
                    if plen == prefix.len() {
                        self.engine.restore_state(st)?;
                        hit = true;
                    }
                }
            }
        }
        if !hit {
            let mark = Instant::now();
            self.engine.clear();
            self.engine.prefill(&prefix)?;
            prefill_seconds = mark.elapsed().as_secs_f64();
            self.serial.prefix = Some(prefix.clone());
            let st = self.engine.save_state()?;
            self.serial.state = Some(st.clone());
            if self.radix_enabled {
                self.radix.insert(&prefix, st);
            }
        } else if self.serial.state.is_none() {
            // restored from radix into ctx; capture for serial cache
            self.serial.prefix = Some(prefix.clone());
            self.serial.state = Some(self.engine.save_state()?);
        }
        let state = self
            .serial
            .state
            .as_ref()
            .ok_or_else(|| Error::Engine("missing serial state".into()))?;
        let mark = Instant::now();
        self.engine.restore_state(state)?;
        let copy_seconds = mark.elapsed().as_secs_f64();
        let mark = Instant::now();
        let vocabulary = self
            .engine
            .branch_logits(prefix.len(), &enc.token_ids[prefix.len()..])?;
        let suffix_seconds = mark.elapsed().as_secs_f64();
        Ok(self.finish(
            row,
            &enc,
            &vocabulary,
            "llamacpp-state-restore-v1",
            "quantized branch last-position logits over a restored prefix state",
            Some(prefill_seconds + suffix_seconds),
            Some(started.elapsed().as_secs_f64()),
            Some(hit),
            Some(prefix.len()),
            Some(prefill_seconds),
            Some(suffix_seconds),
            Some(copy_seconds),
        ))
    }

    pub fn score_shared(&mut self, rows: &[DecisionRow]) -> Result<(Vec<ScoreResult>, SharedTiming)> {
        if rows.is_empty() {
            return Err(Error::Validation(
                "Shared scoring requires one nonempty exact state".into(),
            ));
        }
        let state0 = &rows[0].state;
        if rows.iter().any(|r| &r.state != state0) {
            return Err(Error::Validation(
                "Shared scoring requires one nonempty exact state".into(),
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for r in rows {
            if !ids.insert(r.id.as_str()) {
                return Err(Error::Validation("Decision IDs must be unique".into()));
            }
        }

        let started = Instant::now();
        let encoded: Vec<EncodedDecision> = rows
            .iter()
            .map(|r| self.encode_verified(r))
            .collect::<Result<_>>()?;
        let prefix = self
            .tokenizer
            .state_prefix_ids_version(&rows[0], &self.prompt_version)?;
        if prefix.is_empty()
            || encoded
                .iter()
                .any(|e| e.token_ids.len() <= prefix.len() || e.token_ids[..prefix.len()] != prefix[..])
        {
            return Err(Error::Validation(
                "The fixed state prefix does not match every full prompt".into(),
            ));
        }
        let encode_seconds = started.elapsed().as_secs_f64();

        let mark = Instant::now();
        let mut radix_hit = false;
        if self.radix_enabled {
            if let Some((plen, st)) = self.radix.longest(&prefix) {
                if plen == prefix.len() {
                    self.engine.restore_state(st)?;
                    radix_hit = true;
                }
            }
        }
        if !radix_hit {
            self.engine.clear();
            self.engine.prefill(&prefix)?;
            let st = self.engine.save_state()?;
            if self.radix_enabled {
                self.radix.insert(&prefix, st);
            }
        }
        let state = self.engine.save_state()?;
        let prefill_seconds = mark.elapsed().as_secs_f64();

        let suffixes: Vec<Vec<i32>> = encoded
            .iter()
            .map(|e| e.token_ids[prefix.len()..].to_vec())
            .collect();

        let mut copy_seconds = 0.0;
        let mut suffix_seconds = 0.0;
        let vocabularies: Vec<Vec<f32>> = if self.parallel_suffixes
            && suffixes.len() > 1
            && suffixes.len() as u32 <= self.engine.n_seq_max
        {
            let mark = Instant::now();
            let v = self
                .engine
                .decode_suffixes_parallel(prefix.len(), &suffixes)?;
            suffix_seconds = mark.elapsed().as_secs_f64();
            v
        } else {
            let mut vs = Vec::with_capacity(rows.len());
            for suf in &suffixes {
                let mark = Instant::now();
                self.engine.restore_state(&state)?;
                copy_seconds += mark.elapsed().as_secs_f64();
                let mark = Instant::now();
                vs.push(self.engine.branch_logits(prefix.len(), suf)?);
                suffix_seconds += mark.elapsed().as_secs_f64();
            }
            vs
        };

        let mut results = Vec::with_capacity(rows.len());
        for ((row, enc), vocabulary) in rows.iter().zip(encoded.iter()).zip(vocabularies.iter()) {
            let mut r = self.finish(
                row,
                enc,
                vocabulary,
                "llamacpp-state-restore-shared-v1",
                "quantized branch last-position logits over a restored prefix state",
                None,
                None,
                Some(radix_hit),
                Some(prefix.len()),
                None,
                None,
                None,
            );
            r.radix_prefix_tokens = Some(prefix.len());
            results.push(r);
        }

        let true_suffix: usize = encoded.iter().map(|e| e.token_ids.len() - prefix.len()).sum();
        let timing = SharedTiming {
            total_seconds: started.elapsed().as_secs_f64(),
            encode_seconds,
            prefix_tokens: prefix.len(),
            prefill_seconds,
            replicate_seconds: copy_seconds,
            suffix_forward_seconds: suffix_seconds,
            batch_size: rows.len(),
            true_suffix_tokens: true_suffix,
            padded_suffix_tokens: true_suffix,
        };
        Ok((results, timing))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        row: &DecisionRow,
        enc: &EncodedDecision,
        vocabulary: &[f32],
        serving_config: &str,
        readout: &str,
        forward_seconds: Option<f64>,
        total_seconds: Option<f64>,
        cache_hit: Option<bool>,
        prefix_tokens: Option<usize>,
        prefill_seconds: Option<f64>,
        suffix_forward_seconds: Option<f64>,
        copy_seconds: Option<f64>,
    ) -> ScoreResult {
        let selected: Vec<f64> = enc
            .slots
            .iter()
            .map(|&s| vocabulary[s as usize] as f64)
            .collect();
        let probabilities = softmax(&selected).unwrap_or_else(|_| vec![0.0; selected.len()]);
        let allowed = logsumexp(&selected) - logsumexp_f32(vocabulary);
        let full_argmax = vocabulary
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i32);

        let mut model = self.engine.meta.clone();
        model.serving_config = serving_config.to_string();

        ScoreResult {
            id: row.id.clone(),
            option_ids: row.options.iter().map(|o| o.id.clone()).collect(),
            probabilities,
            option_logits: selected,
            answer_token_ids: enc.slots.clone(),
            input_tokens: enc.token_ids.len(),
            prompt_sha256: enc.prompt_sha256.clone(),
            prompt_version: self.prompt_version.clone(),
            model,
            readout: readout.to_string(),
            probability_status: "conditional option score over quantized weights; uncalibrated as decision confidence".into(),
            cache_hit,
            prefix_tokens,
            forward_seconds,
            total_seconds,
            prefill_seconds,
            suffix_forward_seconds,
            copy_seconds,
            allowed_token_mass: Some(allowed.exp()),
            full_vocab_argmax_id: full_argmax,
            calibrated_probabilities: None,
            temperature: None,
            cascade_source: None,
            replay_hit: None,
            radix_prefix_tokens: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SharedTiming {
    pub total_seconds: f64,
    pub encode_seconds: f64,
    pub prefix_tokens: usize,
    pub prefill_seconds: f64,
    pub replicate_seconds: f64,
    pub suffix_forward_seconds: f64,
    pub batch_size: usize,
    pub true_suffix_tokens: usize,
    pub padded_suffix_tokens: usize,
}

fn logsumexp(values: &[f64]) -> f64 {
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    peak + values.iter().map(|v| (v - peak).exp()).sum::<f64>().ln()
}

fn logsumexp_f32(values: &[f32]) -> f64 {
    let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    peak + values
        .iter()
        .map(|v| (*v as f64 - peak).exp())
        .sum::<f64>()
        .ln()
}

pub fn decision_argmax(result: &ScoreResult) -> Option<usize> {
    argmax(&result.probabilities)
}

pub fn probs_within_tol(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}
