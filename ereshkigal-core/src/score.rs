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
    /// Cached GGUF token ids for answer letters A..=P (load-once).
    letter_token_ids: Option<Vec<i32>>,
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
        // Cache single-letter GGUF token ids once (A..=P) — avoids per-row prompt+letter retokenize.
        let mut letter_token_ids = Vec::with_capacity(16);
        for letter in crate::types::LETTERS.chars().take(16) {
            let s = letter.to_string();
            if let Ok(toks) = engine.gguf_tokenize(&s) {
                if let Some(&last) = toks.last() {
                    letter_token_ids.push(last);
                    continue;
                }
            }
            letter_token_ids.clear();
            break;
        }
        let letter_token_ids = if letter_token_ids.len() == 16 {
            Some(letter_token_ids)
        } else {
            None
        };
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
            letter_token_ids,
            prompt_version: PROMPT_VERSION.to_string(),
            replay_enabled: true,
            radix_enabled: true,
            parallel_suffixes: true,
        }
    }

    pub fn meta(&self) -> &ModelMeta {
        &self.engine.meta
    }

    pub fn embeddings_enabled(&self) -> bool {
        self.engine.embeddings_enabled()
    }

    pub fn n_embd(&self) -> usize {
        self.engine.n_embd()
    }

    /// Score a decision row using a trained [`crate::cosine_head::CosineHead`]
    /// over per-token hidden embeddings extracted via `embeddings_ith`.
    /// Requires the underlying [`EngineOwned`] to have `embeddings: true`.
    pub fn score_cosine(
        &mut self,
        row: &DecisionRow,
        head: &crate::cosine_head::CosineHead,
    ) -> Result<ScoreResult> {
        let started = Instant::now();
        let (prompt, opt_byte_offsets) =
            crate::prompt::render_direct_prompt_with_option_offsets(row)?;
        let (token_ids, offsets) = self.tokenizer.encode_ids_with_offsets(&prompt)?;
        if token_ids.is_empty() {
            return Err(Error::Validation(format!(
                "Row {}: tokenizer produced no tokens",
                row.id
            )));
        }
        let max_limit = if self.max_tokens > 0 {
            self.max_tokens
                .min(self.engine.meta.context_tokens as usize)
        } else {
            self.engine.meta.context_tokens as usize
        };
        if token_ids.len() > max_limit {
            return Err(Error::Validation(format!(
                "Row {}: cosine prompt length {} exceeds limit {}",
                row.id,
                token_ids.len(),
                max_limit
            )));
        }
        let gguf = self.engine.gguf_tokenize(&prompt)?;
        if gguf != token_ids {
            return Err(Error::Validation(format!(
                "Row {}: GGUF tokenization disagrees with the reference tokenizer",
                row.id
            )));
        }
        let opt_positions = crate::tokenizer::locate_token_positions(&offsets, &opt_byte_offsets)
            .ok_or_else(|| {
            Error::Validation(format!(
                "Row {}: could not locate option closing-brace tokens for cosine readout",
                row.id
            ))
        })?;
        if opt_positions.len() != row.options.len() {
            return Err(Error::Validation(format!(
                "Row {}: expected {} option positions, got {}",
                row.id,
                row.options.len(),
                opt_positions.len()
            )));
        }
        let query_pos = token_ids.len() - 1;
        let mut positions = Vec::with_capacity(opt_positions.len() + 1);
        positions.extend_from_slice(&opt_positions);
        positions.push(query_pos);

        let mark = Instant::now();
        let embs = self.engine.token_embeddings_at(&token_ids, &positions)?;
        let forward = mark.elapsed().as_secs_f64();

        let h_query = embs
            .last()
            .ok_or_else(|| Error::Engine("missing query embedding".into()))?;
        let mut option_logits = Vec::with_capacity(row.options.len());
        for opt_emb in &embs[..row.options.len()] {
            let s = head.score_pair(h_query, opt_emb)?;
            option_logits.push(s as f64);
        }
        let probabilities =
            softmax(&option_logits).unwrap_or_else(|_| vec![0.0; option_logits.len()]);
        let answer_token_ids: Vec<i32> = opt_positions.iter().map(|&p| token_ids[p]).collect();

        let mut model = self.engine.meta.clone();
        model.serving_config = "llamacpp-cosine-v1".into();

        Ok(ScoreResult {
            id: row.id.clone(),
            option_ids: row.options.iter().map(|o| o.id.clone()).collect(),
            probabilities,
            option_logits,
            answer_token_ids,
            input_tokens: token_ids.len(),
            prompt_sha256: crate::prompt::digest(&prompt),
            prompt_version: self.prompt_version.clone(),
            model,
            readout: "cosine-ehead-v1".into(),
            probability_status:
                "cosine head option score over hidden embeddings; uncalibrated as decision confidence"
                    .into(),
            cache_hit: Some(false),
            prefix_tokens: None,
            forward_seconds: Some(forward),
            total_seconds: Some(started.elapsed().as_secs_f64()),
            prefill_seconds: None,
            suffix_forward_seconds: None,
            copy_seconds: None,
            allowed_token_mass: None,
            full_vocab_argmax_id: None,
            calibrated_probabilities: None,
            temperature: None,
            cascade_source: None,
            replay_hit: Some(false),
            radix_prefix_tokens: None,
        })
    }

    #[cfg(feature = "vision")]
    pub fn load_vision(
        &self,
        cfg: &crate::vision::VisionConfig,
    ) -> Result<crate::vision::VisionContext> {
        crate::vision::VisionContext::load(cfg, &self.engine)
    }

    #[cfg(feature = "vision")]
    pub fn score_visual(
        &mut self,
        vctx: &crate::vision::VisionContext,
        png_paths: &[&std::path::Path],
        row: &DecisionRow,
    ) -> Result<ScoreResult> {
        let started = Instant::now();
        let slots = self.tokenizer.slot_ids(row.options.len())?;
        let mark = Instant::now();
        let (vocabulary, total_tokens, prompt_sha256) =
            crate::vision::eval_visual_vocab_with_ctx(&mut self.engine, vctx, png_paths, row)?;
        let forward = mark.elapsed().as_secs_f64();
        let enc = EncodedDecision {
            token_ids: vec![0; total_tokens],
            slots,
            prompt: String::new(),
            prompt_sha256,
        };
        Ok(self.finish(
            row,
            &enc,
            &vocabulary,
            "llamacpp-vision-mtmd-v1",
            "mtmd/mmproj image-conditioned last-position letter logits",
            Some(forward),
            Some(started.elapsed().as_secs_f64()),
            Some(false),
            None,
            None,
            None,
            None,
        ))
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
        let n = enc.slots.len();
        let mut letter_ids: Vec<i32> = if let Some(cached) = &self.letter_token_ids {
            cached.iter().copied().take(n).collect()
        } else {
            Vec::new()
        };
        if letter_ids.len() != n || slots_match_letters(&enc.slots, &letter_ids).is_err() {
            // Fall back to prompt+letter last-token (BPE context may differ from lone letter).
            letter_ids.clear();
            for letter in crate::types::LETTERS.chars().take(n) {
                let with = self
                    .engine
                    .gguf_tokenize(&format!("{}{letter}", enc.prompt))?;
                let last = with.last().copied().ok_or_else(|| {
                    Error::Validation(format!("GGUF produced no token for letter {letter}"))
                })?;
                letter_ids.push(last);
            }
        }
        slots_match_letters(&enc.slots, &letter_ids)?;
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
            let v = match self
                .engine
                .decode_suffixes_parallel(prefix.len(), &suffixes)
            {
                Ok(v) => v,
                Err(e) if crate::engine::EngineOwned::error_is_kv_slot(&e) => {
                    // Single clear-and-retry: wipe every seq, restore prefix, try
                    // once more before the caller falls back to heuristic.
                    self.engine.clear_all_seqs()?;
                    self.engine.restore_state(&state)?;
                    self.engine
                        .decode_suffixes_parallel(prefix.len(), &suffixes)?
                }
                Err(e) => return Err(e),
            };
            suffix_seconds = mark.elapsed().as_secs_f64();
            v
        } else {
            let mut vs = Vec::with_capacity(rows.len());
            for suf in &suffixes {
                let mark = Instant::now();
                self.engine.restore_state(&state)?;
                copy_seconds += mark.elapsed().as_secs_f64();
                let mark = Instant::now();
                let logits = match self.engine.branch_logits(prefix.len(), suf) {
                    Ok(v) => v,
                    Err(e) if crate::engine::EngineOwned::error_is_kv_slot(&e) => {
                        self.engine.clear_all_seqs()?;
                        self.engine.restore_state(&state)?;
                        self.engine.branch_logits(prefix.len(), suf)?
                    }
                    Err(e) => return Err(e),
                };
                vs.push(logits);
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

    /// Pack all `rows` sharing one exact `state` into a single `joint-options-v1`
    /// prompt (`STATE + FIELD 1..Q`), locate every `ANSWER:` colon token offset,
    /// and decode all fields in one forward pass. Falls back gracefully to
    /// [`Self::score_shared`] if token positions cannot be located or the prompt
    /// exceeds token/context limits.
    pub fn score_joint_prompt(
        &mut self,
        rows: &[DecisionRow],
    ) -> Result<(Vec<ScoreResult>, SharedTiming)> {
        if rows.is_empty() {
            return Err(Error::Validation(
                "Joint scoring requires one nonempty exact state".into(),
            ));
        }
        let state0 = &rows[0].state;
        if rows.iter().any(|r| &r.state != state0) {
            return Err(Error::Validation(
                "Joint scoring requires one nonempty exact state".into(),
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for r in rows {
            if !ids.insert(r.id.as_str()) {
                return Err(Error::Validation("Decision IDs must be unique".into()));
            }
        }

        let started = Instant::now();
        let (prompt, colon_byte_offsets) =
            match crate::prompt::render_joint_batch_prompt_with_offsets(state0, rows) {
                Ok(v) => v,
                Err(_) => return self.score_shared(rows),
            };
        let (token_ids, offsets) = match self.tokenizer.encode_ids_with_offsets(&prompt) {
            Ok(v) => v,
            Err(_) => return self.score_shared(rows),
        };
        let ctx_limit = self.engine.meta.context_tokens as usize;
        let max_limit = if self.max_tokens > 0 {
            self.max_tokens.min(ctx_limit)
        } else {
            ctx_limit
        };
        if token_ids.is_empty() || token_ids.len() > max_limit {
            return self.score_shared(rows);
        }
        let gguf_ids = match self.engine.gguf_tokenize(&prompt) {
            Ok(v) => v,
            Err(_) => return self.score_shared(rows),
        };
        if gguf_ids != token_ids {
            return self.score_shared(rows);
        }
        let Some(read_positions) =
            crate::tokenizer::locate_token_positions(&offsets, &colon_byte_offsets)
        else {
            return self.score_shared(rows);
        };
        if read_positions.len() != rows.len()
            || read_positions.iter().any(|&p| p >= token_ids.len())
        {
            return self.score_shared(rows);
        }

        let mut all_slots = Vec::with_capacity(rows.len());
        for row in rows {
            let _ = letter_gbnf(row.options.len())?;
            let slots = self.tokenizer.slot_ids(row.options.len())?;
            if let Some(cached) = &self.letter_token_ids {
                let letter_ids: Vec<i32> = cached.iter().copied().take(slots.len()).collect();
                if slots_match_letters(&slots, &letter_ids).is_err() {
                    return self.score_shared(rows);
                }
            }
            all_slots.push(slots);
        }

        let encode_seconds = started.elapsed().as_secs_f64();
        let mark = Instant::now();
        let vocabularies = self
            .engine
            .decode_logits_at_positions(&token_ids, &read_positions)?;
        let forward_seconds = mark.elapsed().as_secs_f64();

        let prompt_sha256 = crate::prompt::digest(&prompt);
        let prefix_tokens = read_positions.first().copied().unwrap_or(0);
        let mut results = Vec::with_capacity(rows.len());
        for ((row, slots), vocabulary) in rows
            .iter()
            .zip(all_slots.into_iter())
            .zip(vocabularies.iter())
        {
            let enc = EncodedDecision {
                token_ids: token_ids.clone(),
                slots,
                prompt: prompt.clone(),
                prompt_sha256: prompt_sha256.clone(),
            };
            let mut r = self.finish(
                row,
                &enc,
                vocabulary,
                "llamacpp-joint-options-v1",
                "quantized multi-field ANSWER: colon logits in one forward pass",
                Some(forward_seconds),
                None,
                Some(false),
                Some(prefix_tokens),
                Some(forward_seconds),
                Some(0.0),
                Some(0.0),
            );
            r.prompt_version = crate::types::PROMPT_VERSION_JOINT.to_string();
            results.push(r);
        }

        let suffix_tokens = token_ids.len().saturating_sub(prefix_tokens);
        let timing = SharedTiming {
            total_seconds: started.elapsed().as_secs_f64(),
            encode_seconds,
            prefix_tokens,
            prefill_seconds: forward_seconds,
            replicate_seconds: 0.0,
            suffix_forward_seconds: 0.0,
            batch_size: rows.len(),
            true_suffix_tokens: suffix_tokens,
            padded_suffix_tokens: suffix_tokens,
        };
        Ok((results, timing))
    }

    /// Confidence-gated AutoThink scoring: compute `p0` via [`Self::score_direct`],
    /// check [`crate::autothink::should_think`], and if triggered generate bounded
    /// thought tokens, append them to the prompt, re-score `p1`, fuse `(1-α)p0 + αp1`,
    /// and optionally apply post-think temperature scaling.
    pub fn score_autothink(
        &mut self,
        row: &DecisionRow,
        cfg: &crate::autothink::ThinkConfig,
        conformal_set_size: usize,
        post_think_t: Option<f64>,
    ) -> Result<(ScoreResult, crate::autothink::AutoThinkOutcome)> {
        self.score_autothink_inner(row, cfg, conformal_set_size, None, post_think_t)
    }

    /// Variant of [`Self::score_autothink`] that computes conformal set size from
    /// `p0` and an optional fitted `qhat` without an extra forward pass.
    pub fn score_autothink_conformal(
        &mut self,
        row: &DecisionRow,
        cfg: &crate::autothink::ThinkConfig,
        qhat: Option<f64>,
        post_think_t: Option<f64>,
    ) -> Result<(ScoreResult, crate::autothink::AutoThinkOutcome)> {
        self.score_autothink_inner(row, cfg, 1, qhat, post_think_t)
    }

    fn score_autothink_inner(
        &mut self,
        row: &DecisionRow,
        cfg: &crate::autothink::ThinkConfig,
        conformal_set_size: usize,
        qhat: Option<f64>,
        post_think_t: Option<f64>,
    ) -> Result<(ScoreResult, crate::autothink::AutoThinkOutcome)> {
        let mut res0 = self.score_direct(row)?;
        let p0 = res0.probabilities.clone();
        let max_p0 = p0.iter().copied().fold(0.0_f64, f64::max);
        let eff_csize = qhat
            .map(|q| crate::option_set(&p0, q).len())
            .unwrap_or(conformal_set_size);
        if !crate::autothink::should_think(cfg, eff_csize, max_p0) {
            return Ok((
                res0,
                crate::autothink::AutoThinkOutcome {
                    probs: p0,
                    think_invoked: false,
                    think_tokens: 0,
                    p0_vs_p1_flipped: false,
                },
            ));
        }

        let enc = self.encode_verified(row)?;
        const THINK_OFF_SUFFIX: &str = "\n</think>\n\n";
        let (seed_prompt, close_suffix) =
            if let Some(stripped) = enc.prompt.strip_suffix(THINK_OFF_SUFFIX) {
                (stripped.to_string(), THINK_OFF_SUFFIX)
            } else {
                (format!("{}<think>\n", enc.prompt), THINK_OFF_SUFFIX)
            };
        let seed_tokens = self.engine.gguf_tokenize(&seed_prompt)?;
        let (thought_tokens, think_tokens) =
            crate::autothink::generate_thought(&mut self.engine, &seed_tokens, cfg.budget)?;

        let (p1, p1_logits) = if thought_tokens.is_empty() {
            (p0.clone(), res0.option_logits.clone())
        } else {
            let close_tokens = self.engine.gguf_tokenize(close_suffix)?;
            let max_ctx = self.engine.meta.context_tokens as usize;
            let avail_thought = max_ctx.saturating_sub(seed_tokens.len() + close_tokens.len());
            let mut p1_tokens =
                Vec::with_capacity(seed_tokens.len() + thought_tokens.len().min(avail_thought) + close_tokens.len());
            p1_tokens.extend_from_slice(&seed_tokens);
            p1_tokens.extend_from_slice(&thought_tokens[..thought_tokens.len().min(avail_thought)]);
            p1_tokens.extend_from_slice(&close_tokens);
            let p1_vocab = self.engine.full_logits(&p1_tokens)?;
            let logits: Vec<f64> = enc
                .slots
                .iter()
                .map(|&s| p1_vocab[s as usize] as f64)
                .collect();
            let probs = softmax(&logits).unwrap_or_else(|_| vec![0.0; logits.len()]);
            (probs, logits)
        };

        let outcome = crate::autothink::build_autothink_outcome(
            &p0,
            &p1,
            think_tokens,
            cfg.alpha,
            post_think_t,
        )?;
        res0.probabilities = outcome.probs.clone();
        res0.option_logits = p1_logits;
        res0.model.serving_config = "llamacpp-autothink-v1".into();
        res0.readout = "confidence-gated AutoThink fused p0/p1 option logits".into();
        if let Some(t) = post_think_t {
            if t.is_finite() && t > 0.0 && (t - 1.0).abs() > 1e-12 {
                res0.calibrated_probabilities = Some(outcome.probs.clone());
                res0.temperature = Some(t);
            }
        }
        Ok((res0, outcome))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine_head::CosineHead;
    use crate::engine::EngineConfig;
    use crate::types::OptionSpec;
    use serde_json::json;
    use std::path::PathBuf;

    /// Live GGUF smoke for `EngineConfig.embeddings = true` + `token_embeddings_at`
    /// + `Scorer::score_cosine`. Skips cleanly when `ERESHKIGAL_GGUF` is unset.
    #[test]
    fn score_cosine_live_gguf_smoke() {
        let Ok(gguf) = std::env::var("ERESHKIGAL_GGUF") else {
            return;
        };
        let path = PathBuf::from(&gguf);
        if !path.is_file() {
            return;
        }
        let tok_src = std::env::var("ERESHKIGAL_TOKENIZER_SOURCE")
            .unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
        let tok_rev = std::env::var("ERESHKIGAL_TOKENIZER_REVISION").unwrap_or_else(|_| {
            "c1899de289a04d12100db370d81485cdf75e47ca".into()
        });
        let (engine, tokenizer) = EngineOwned::load(EngineConfig {
            gguf: path,
            tokenizer_source: tok_src,
            tokenizer_revision: tok_rev,
            max_prompt_tokens: 1024,
            threads: 4,
            n_gpu_layers: 0,
            n_seq_max: 1,
            embeddings: true,
            adapter: None,
            adapter_scale: 1.0,
        })
        .expect("load embeddings engine");
        let n_embd = engine.n_embd();
        assert!(n_embd > 0);
        let mut scorer = Scorer::new(engine, tokenizer);
        let head = CosineHead::random_init(n_embd, 13);
        let row = DecisionRow {
            id: "cos-smoke".into(),
            state: json!("All tests passed and latency is 4ms."),
            question: "Should we promote?".into(),
            options: vec![
                OptionSpec {
                    id: "promote".into(),
                    description: "Promote the build.".into(),
                },
                OptionSpec {
                    id: "hold".into(),
                    description: "Hold the build.".into(),
                },
            ],
        };
        let res = scorer.score_cosine(&row, &head).expect("score_cosine");
        assert_eq!(res.option_ids, vec!["promote", "hold"]);
        assert_eq!(res.probabilities.len(), 2);
        assert!(res.probabilities.iter().all(|p| p.is_finite()));
        assert!((res.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-5);
        assert_eq!(res.readout, "cosine-ehead-v1");
        assert_eq!(res.model.serving_config, "llamacpp-cosine-v1");
    }
}
