use crate::error::{Error, Result};
use crate::tokenizer::ReferenceTokenizer;
use crate::types::{GgufMeta, ModelMeta};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::session::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use sha2::{Digest, Sha256};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::OnceLock;

const DECODE_CHUNK: usize = 512;
/// Per-seq suffix KV headroom beyond the shared prefix. Option labels in
/// Wordkeep `semantic_decide` / adaptive decrees routinely exceed 64 tokens
/// (criteria + rationale); undersizing here surfaces as llama.cpp
/// `NoKvCacheSlot` and forces heuristic fallback.
pub const DEFAULT_MAX_SUFFIX_TOKENS: u32 = 256;

/// llama.cpp allows one `llama_backend_init` per process. Wordkeep tandem (GPU
/// draft + CPU draft) and draft+verify need multiple models, so we leak the
/// first successful backend and reuse it.
static SHARED_BACKEND: OnceLock<std::result::Result<&'static LlamaBackend, String>> =
    OnceLock::new();

fn shared_backend() -> Result<&'static LlamaBackend> {
    let cached = SHARED_BACKEND.get_or_init(|| match LlamaBackend::init() {
        Ok(backend) => Ok(Box::leak(Box::new(backend)) as &'static LlamaBackend),
        Err(e) => Err(e.to_string()),
    });
    match cached {
        Ok(b) => Ok(*b),
        Err(e) => Err(Error::Engine(e.clone())),
    }
}

#[derive(Clone)]
pub struct EngineConfig {
    pub gguf: PathBuf,
    pub tokenizer_source: String,
    pub tokenizer_revision: String,
    pub max_prompt_tokens: usize,
    pub threads: i32,
    pub n_gpu_layers: u32,
    pub n_seq_max: u32,
    pub embeddings: bool,
    pub adapter: Option<PathBuf>,
    /// LoRA adapter strength passed to `lora_adapter_set` (default 1.0).
    pub adapter_scale: f32,
}

/// Process-scoped llama.cpp scorer. The model is leaked for a `'static` context lifetime
/// (appropriate for a CLI or long-lived scoring process). Backend is process-global
/// ([`shared_backend`]) so tandem GPU+CPU and draft+verify can coexist.
pub struct EngineOwned {
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    pub meta: ModelMeta,
    pub gguf_path: PathBuf,
    pub n_seq_max: u32,
    embeddings_enabled: bool,
}

unsafe impl Send for EngineOwned {}

impl EngineOwned {
    pub fn load(cfg: EngineConfig) -> Result<(Self, ReferenceTokenizer)> {
        if !cfg.gguf.is_file() {
            return Err(Error::Engine(format!(
                "GGUF checkpoint not found: {}",
                cfg.gguf.display()
            )));
        }
        let tokenizer =
            ReferenceTokenizer::from_hub(&cfg.tokenizer_source, &cfg.tokenizer_revision)?;
        let gguf_meta = hash_gguf(&cfg.gguf)?;

        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        let backend = shared_backend()?;
        let mut model_params = LlamaModelParams::default();
        if cfg.n_gpu_layers > 0 {
            model_params = model_params.with_n_gpu_layers(cfg.n_gpu_layers);
        }
        let model_params = pin!(model_params);
        let model = LlamaModel::load_from_file(backend, &cfg.gguf, &model_params)
            .map_err(|e| Error::Engine(format!("load GGUF: {e}")))?;
        let model: &'static LlamaModel = Box::leak(Box::new(model));

        let n_seq_max = cfg.n_seq_max.max(1);
        // Parallel suffix decode needs prefix + n_seq_max * suffix headroom (NoKvCacheSlot).
        let max_suffix = DEFAULT_MAX_SUFFIX_TOKENS;
        let ctx_tokens = required_n_ctx(cfg.max_prompt_tokens as u32, n_seq_max, max_suffix);
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_tokens))
            .with_n_threads(cfg.threads)
            .with_n_threads_batch(cfg.threads)
            .with_n_seq_max(n_seq_max)
            .with_embeddings(cfg.embeddings);
        if cfg.embeddings {
            // Per-token hidden states via llama_get_embeddings_ith require POOLING_TYPE_NONE.
            ctx_params = ctx_params.with_pooling_type(LlamaPoolingType::None);
        }

        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Engine(format!("create context: {e}")))?;

        if let Some(adapter_path) = &cfg.adapter {
            let scale = if cfg.adapter_scale.is_finite() && cfg.adapter_scale > 0.0 {
                cfg.adapter_scale
            } else {
                1.0
            };
            let mut adapter = model
                .lora_adapter_init(adapter_path)
                .map_err(|e| Error::Engine(format!("lora_adapter_init: {e}")))?;
            ctx.lora_adapter_set(&mut adapter, scale)
                .map_err(|e| Error::Engine(format!("lora_adapter_set: {e}")))?;
            Box::leak(Box::new(adapter));
        }

        verify_vocab_agreement(&tokenizer, model)?;

        let meta = ModelMeta {
            source: cfg.tokenizer_source,
            revision: cfg.tokenizer_revision,
            backend: "llamacpp".into(),
            dtype: "gguf-quantized".into(),
            gguf: gguf_meta,
            vocab_size: model.n_vocab(),
            threads: cfg.threads,
            n_gpu_layers: cfg.n_gpu_layers as i32,
            max_prompt_tokens: cfg.max_prompt_tokens,
            context_tokens: ctx.n_ctx(),
            decode_chunk: DECODE_CHUNK,
            serving_config: "unset".into(),
        };

        Ok((
            Self {
                model,
                ctx,
                meta,
                gguf_path: cfg.gguf,
                n_seq_max,
                embeddings_enabled: cfg.embeddings,
            },
            tokenizer,
        ))
    }

    pub fn clear(&mut self) {
        self.ctx.clear_kv_cache();
    }

    /// Drop every sequence slot (`0..n_seq_max`) so parallel/adaptive waves
    /// never leak occupied KV cells into the next restore/copy.
    pub fn clear_all_seqs(&mut self) -> Result<()> {
        for seq in 0..self.n_seq_max {
            let ok = self
                .ctx
                .clear_kv_cache_seq(Some(seq), None, None)
                .map_err(|e| Error::Engine(e.to_string()))?;
            if !ok {
                // Fall back to a full wipe once rather than leaving a half-cleared cache.
                self.clear();
                return Ok(());
            }
        }
        Ok(())
    }

    fn is_kv_slot_error(msg: &str) -> bool {
        let lower = msg.to_ascii_lowercase();
        lower.contains("nokvcacheslot")
            || lower.contains("no kv cache slot")
            || lower.contains("failed to find a kv slot")
            || lower.contains("kv cache is full")
    }

    pub fn full_logits(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.clear();
        self.decode_logits_seq(tokens, 0, 0)
    }

    pub fn prefill(&mut self, prefix: &[i32]) -> Result<()> {
        let _ = self.decode_logits_seq(prefix, 0, 0)?;
        Ok(())
    }

    pub fn save_state(&self) -> Result<SeqState> {
        self.ctx
            .state_seq_get(0, LlamaStateSeqFlags::empty())
            .map_err(|e| Error::Engine(format!("state_seq_get: {e}")))
    }

    pub fn restore_state(&mut self, state: &SeqState) -> Result<()> {
        // Clear *all* seq slots first — leftover parallel copies on 1..n-1
        // otherwise keep KV cells occupied and the next wave hits NoKvCacheSlot.
        self.clear_all_seqs()?;
        self.ctx
            .state_seq_set(state, 0)
            .map_err(|e| Error::Engine(format!("state_seq_set: {e}")))
    }

    pub fn branch_logits(&mut self, prefix_len: usize, suffix: &[i32]) -> Result<Vec<f32>> {
        self.decode_logits_seq(suffix, prefix_len as i32, 0)
    }

    /// Copy KV from seq 0 onto `dest` (parallel suffixes).
    pub fn copy_seq(&mut self, src: i32, dest: i32) -> Result<()> {
        self.ctx
            .copy_kv_cache_seq(src, dest, None, None)
            .map_err(|e| Error::Engine(format!("copy_kv_cache_seq: {e}")))
    }

    /// Decode several suffixes in one batch after prefix lives on seq 0.
    /// Copies seq 0 → 0..n-1 then decodes each suffix on its seq id.
    pub fn decode_suffixes_parallel(
        &mut self,
        prefix_len: usize,
        suffixes: &[Vec<i32>],
    ) -> Result<Vec<Vec<f32>>> {
        if suffixes.is_empty() {
            return Err(Error::Engine("no suffixes to decode".into()));
        }
        if suffixes.len() as u32 > self.n_seq_max {
            return Err(Error::Engine(format!(
                "batch of {} suffixes exceeds n_seq_max {}",
                suffixes.len(),
                self.n_seq_max
            )));
        }
        // Cap each suffix to the reserved headroom so we never request more KV
        // cells than `required_n_ctx` budgeted at load time.
        let capped: Vec<Vec<i32>> = suffixes
            .iter()
            .map(|s| {
                if s.len() as u32 > DEFAULT_MAX_SUFFIX_TOKENS {
                    truncate_tokens_head_tail(s, DEFAULT_MAX_SUFFIX_TOKENS as usize)
                } else {
                    s.clone()
                }
            })
            .collect();
        self.decode_suffixes_parallel_inner(prefix_len, &capped)
    }

    /// True when an engine error looks like llama.cpp KV slot exhaustion.
    pub fn error_is_kv_slot(err: &Error) -> bool {
        Self::is_kv_slot_error(&err.to_string())
    }

    fn decode_suffixes_parallel_inner(
        &mut self,
        prefix_len: usize,
        suffixes: &[Vec<i32>],
    ) -> Result<Vec<Vec<f32>>> {
        // Drop stale copies on 1..n_seq_max-1 before re-branching from seq 0.
        for seq in 1..self.n_seq_max {
            let _ = self.ctx.clear_kv_cache_seq(Some(seq), None, None);
        }
        // Caller must prefill/restore prefix on seq 0; do not clear seq 0 here.
        for (i, suf) in suffixes.iter().enumerate() {
            if suf.is_empty() {
                return Err(Error::Engine(format!("empty suffix at {i}")));
            }
            if i > 0 {
                self.copy_seq(0, i as i32)?;
            }
        }
        let n_tokens: usize = suffixes.iter().map(|s| s.len()).sum();
        let mut batch = LlamaBatch::new(n_tokens.max(1), self.n_seq_max as i32);
        let mut last_offsets: Vec<i32> = Vec::with_capacity(suffixes.len());
        for (seq, suf) in suffixes.iter().enumerate() {
            for (i, &tok) in suf.iter().enumerate() {
                let pos = prefix_len as i32 + i as i32;
                let want = i + 1 == suf.len();
                batch
                    .add(LlamaToken(tok), pos, &[seq as i32], want)
                    .map_err(|e| Error::Engine(format!("batch.add: {e}")))?;
                if want {
                    last_offsets.push(batch.n_tokens() - 1);
                }
            }
        }
        self.ctx
            .decode(&mut batch)
            .map_err(|e| Error::Engine(format!("parallel llama_decode: {e}")))?;
        let n_vocab = self.model.n_vocab() as usize;
        let mut out = Vec::with_capacity(suffixes.len());
        for off in last_offsets {
            let logits = self.ctx.get_logits_ith(off);
            if logits.len() < n_vocab {
                return Err(Error::Engine("short parallel logits".into()));
            }
            out.push(logits[..n_vocab].to_vec());
        }
        Ok(out)
    }

    /// Decode `tokens` from position 0 on seq 0 and extract vocabulary logits at
    /// each requested token index in `read_positions` via `get_logits_ith`.
    /// When `tokens.len() <= ctx.n_batch()`, submits all tokens in one `LlamaBatch`.
    pub fn decode_logits_at_positions(
        &mut self,
        tokens: &[i32],
        read_positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::Engine(
                "Refusing to decode an empty token list".into(),
            ));
        }
        if read_positions.is_empty() {
            return Err(Error::Engine(
                "Refusing to decode with empty read_positions".into(),
            ));
        }
        let total = tokens.len();
        let mut want_mask = vec![false; total];
        for &pos in read_positions {
            if pos >= total {
                return Err(Error::Engine(format!(
                    "read_position {pos} out of bounds for {total} tokens"
                )));
            }
            want_mask[pos] = true;
        }

        self.clear();
        let n_vocab = self.model.n_vocab() as usize;
        let chunk_cap = (self.ctx.n_batch() as usize).max(1).min(total);
        let mut batch = LlamaBatch::new(chunk_cap, self.n_seq_max.max(1) as i32);
        let mut out: Vec<Option<Vec<f32>>> = vec![None; read_positions.len()];

        for offset in (0..total).step_by(chunk_cap) {
            let end = (offset + chunk_cap).min(total);
            let chunk = &tokens[offset..end];
            batch.clear();
            for (i, &tok) in chunk.iter().enumerate() {
                let global_pos = offset + i;
                batch
                    .add(
                        LlamaToken(tok),
                        global_pos as i32,
                        &[0],
                        want_mask[global_pos],
                    )
                    .map_err(|e| Error::Engine(format!("batch.add: {e}")))?;
            }
            self.ctx.decode(&mut batch).map_err(|e| {
                Error::Engine(format!(
                    "joint llama_decode failed; raise --max-tokens if prompts grew: {e}"
                ))
            })?;
            for (slot_idx, &pos) in read_positions.iter().enumerate() {
                if pos >= offset && pos < end {
                    let local_i = (pos - offset) as i32;
                    let logits = self.ctx.get_logits_ith(local_i);
                    if logits.len() < n_vocab {
                        return Err(Error::Engine(
                            "llama.cpp returned short joint logits".into(),
                        ));
                    }
                    out[slot_idx] = Some(logits[..n_vocab].to_vec());
                }
            }
        }

        out.into_iter()
            .enumerate()
            .map(|(i, v)| {
                v.ok_or_else(|| {
                    Error::Engine(format!("missing logits for read_position index {i}"))
                })
            })
            .collect()
    }

    /// Model's end-of-sequence token id.
    pub fn eos_token(&self) -> i32 {
        self.model.vocab().eos().0
    }

    /// Whether `token` is an end-of-generation token (EOS, EOT, etc.).
    pub fn is_eog_token(&self, token: i32) -> bool {
        self.model.vocab().is_eog(LlamaToken(token))
    }

    /// Single-token id for `"</think>"` when the model vocabulary encodes it as one token.
    pub fn think_end_token(&self) -> Option<i32> {
        self.gguf_tokenize("</think>")
            .ok()
            .and_then(|v| (v.len() == 1).then_some(v[0]))
    }

    pub fn gguf_tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let tokens = self.model.vocab().tokenize(text.as_bytes(), false, true);
        Ok(tokens.into_iter().map(|t| t.0).collect())
    }

    pub fn last_embedding(&self) -> Result<Vec<f32>> {
        self.ctx
            .embeddings_ith(-1)
            .or_else(|_| self.ctx.embeddings_ith(0))
            .map(|s| s.to_vec())
            .map_err(|e| Error::Engine(format!("embeddings: {e}")))
    }

    pub fn embeddings_enabled(&self) -> bool {
        self.embeddings_enabled
    }

    pub fn n_embd(&self) -> usize {
        self.model.n_embd() as usize
    }

    pub fn n_batch(&self) -> u32 {
        self.ctx.n_batch()
    }

    #[cfg(feature = "vision")]
    pub(crate) fn llama_model(&self) -> &'static LlamaModel {
        self.model
    }

    #[cfg(feature = "vision")]
    pub(crate) fn llama_context(&self) -> &LlamaContext<'static> {
        &self.ctx
    }

    #[cfg(feature = "vision")]
    pub(crate) fn last_vocab_logits(&self) -> Result<Vec<f32>> {
        let n_vocab = self.model.n_vocab() as usize;
        let logits = self.ctx.get_logits();
        if logits.len() < n_vocab {
            return Err(Error::Engine("llama.cpp returned short logits".into()));
        }
        Ok(logits[..n_vocab].to_vec())
    }

    /// Decode `tokens` from position 0 on seq 0 and extract per-token hidden
    /// embeddings at each requested token index in `positions` via `embeddings_ith`.
    /// Requires the engine to have been loaded with `EngineConfig.embeddings = true`.
    pub fn token_embeddings_at(
        &mut self,
        tokens: &[i32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if !self.embeddings_enabled {
            return Err(Error::Engine(
                "embeddings not enabled on EngineOwned (set EngineConfig.embeddings = true)".into(),
            ));
        }
        if tokens.is_empty() {
            return Err(Error::Engine(
                "Refusing to decode an empty token list".into(),
            ));
        }
        if positions.is_empty() {
            return Err(Error::Engine(
                "Refusing to extract embeddings with empty positions".into(),
            ));
        }
        let total = tokens.len();
        let mut want_mask = vec![false; total];
        for &pos in positions {
            if pos >= total {
                return Err(Error::Engine(format!(
                    "embedding position {pos} out of bounds for {total} tokens"
                )));
            }
            want_mask[pos] = true;
        }

        self.clear();
        let chunk_cap = (self.ctx.n_batch() as usize).max(1).min(total);
        let mut batch = LlamaBatch::new(chunk_cap, self.n_seq_max.max(1) as i32);
        let mut out: Vec<Option<Vec<f32>>> = vec![None; positions.len()];

        for offset in (0..total).step_by(chunk_cap) {
            let end = (offset + chunk_cap).min(total);
            let chunk = &tokens[offset..end];
            batch.clear();
            for (i, &tok) in chunk.iter().enumerate() {
                let global_pos = offset + i;
                batch
                    .add(
                        LlamaToken(tok),
                        global_pos as i32,
                        &[0],
                        want_mask[global_pos],
                    )
                    .map_err(|e| Error::Engine(format!("batch.add: {e}")))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| Error::Engine(format!("embeddings llama_decode failed: {e}")))?;
            for (slot_idx, &pos) in positions.iter().enumerate() {
                if pos >= offset && pos < end {
                    let local_i = (pos - offset) as i32;
                    let emb = self
                        .ctx
                        .embeddings_ith(local_i)
                        .map_err(|e| Error::Engine(format!("embeddings_ith({local_i}): {e}")))?;
                    out[slot_idx] = Some(emb.to_vec());
                }
            }
        }

        out.into_iter()
            .enumerate()
            .map(|(i, v)| {
                v.ok_or_else(|| Error::Engine(format!("missing embedding for position index {i}")))
            })
            .collect()
    }

    pub fn gguf_chat_template(&self) -> Result<String> {
        let t = self
            .model
            .chat_template(None)
            .map_err(|e| Error::Engine(format!("chat_template: {e}")))?;
        Ok(t.as_c_str().to_string_lossy().into_owned())
    }

    pub fn render_minijinja(&self, template: &str, messages: &[(String, String)]) -> Result<String> {
        let env = minijinja::Environment::new();
        let tmpl = env
            .template_from_str(template)
            .map_err(|e| Error::Engine(format!("minijinja parse: {e}")))?;
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|(r, c)| serde_json::json!({"role": r, "content": c}))
            .collect();
        tmpl.render(serde_json::json!({
            "messages": msgs,
            "add_generation_prompt": true,
            "enable_thinking": false,
        }))
            .map_err(|e| Error::Engine(format!("minijinja render: {e}")))
    }

    fn decode_logits_seq(&mut self, tokens: &[i32], start: i32, seq: i32) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::Engine(
                "Refusing to decode an empty token list".into(),
            ));
        }
        // Prefill / full_logits (start == 0): deterministically head+tail truncate
        // above n_ctx, matching decode_suffixes_parallel's KV safety fence.
        let owned = if start == 0 {
            let n_ctx = self.ctx.n_ctx() as usize;
            if n_ctx > 0 && tokens.len() > n_ctx {
                Some(truncate_tokens_head_tail(tokens, n_ctx))
            } else {
                None
            }
        } else {
            None
        };
        let tokens = owned.as_deref().unwrap_or(tokens);
        match self.decode_logits_seq_inner(tokens, start, seq) {
            Ok(v) => Ok(v),
            Err(e) if Self::is_kv_slot_error(&e.to_string()) => {
                // Single clear_all_seqs retry, matching parallel-suffix recovery.
                self.clear_all_seqs()?;
                self.decode_logits_seq_inner(tokens, start, seq)
            }
            Err(e) => Err(e),
        }
    }

    fn decode_logits_seq_inner(
        &mut self,
        tokens: &[i32],
        start: i32,
        seq: i32,
    ) -> Result<Vec<f32>> {
        let n_vocab = self.model.n_vocab() as usize;
        let mut batch = LlamaBatch::new(DECODE_CHUNK, self.n_seq_max.max(1) as i32);
        let total = tokens.len();
        for offset in (0..total).step_by(DECODE_CHUNK) {
            let end = (offset + DECODE_CHUNK).min(total);
            let chunk = &tokens[offset..end];
            batch.clear();
            for (i, &tok) in chunk.iter().enumerate() {
                let pos = start + offset as i32 + i as i32;
                let want_logits = offset + i == total - 1;
                batch
                    .add(LlamaToken(tok), pos, &[seq], want_logits)
                    .map_err(|e| Error::Engine(format!("batch.add: {e}")))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| Error::Engine(format!(
                    "llama_decode failed; raise --max-tokens if prompts grew: {e}"
                )))?;
        }
        let logits = self.ctx.get_logits();
        if logits.len() < n_vocab {
            return Err(Error::Engine("llama.cpp returned short logits".into()));
        }
        Ok(logits[..n_vocab].to_vec())
    }
}

fn hash_gguf(path: &Path) -> Result<GgufMeta> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let sha256 = hex::encode(hasher.finalize());
    let meta = std::fs::metadata(path)?;
    Ok(GgufMeta {
        file: path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.gguf")
            .to_string(),
        bytes: meta.len(),
        sha256,
    })
}

fn verify_vocab_agreement(tokenizer: &ReferenceTokenizer, model: &LlamaModel) -> Result<()> {
    use crate::types::{DecisionRow, OptionSpec};
    use serde_json::json;
    let row = DecisionRow {
        id: "vocabulary-probe".into(),
        state: json!("probe evidence"),
        question: "probe criterion?".into(),
        options: vec![
            OptionSpec {
                id: "yes".into(),
                description: "Yes.".into(),
            },
            OptionSpec {
                id: "no".into(),
                description: "No.".into(),
            },
        ],
    };
    let prompt = crate::prompt::render_prompt(&row)?;
    let reference = tokenizer.encode_ids(&prompt)?;
    let gguf_ids: Vec<i32> = model
        .vocab()
        .tokenize(prompt.as_bytes(), false, true)
        .into_iter()
        .map(|t| t.0)
        .collect();
    if gguf_ids != reference {
        return Err(Error::Engine(
            "The GGUF vocabulary disagrees with the reference tokenizer".into(),
        ));
    }
    for letter in crate::types::LETTERS.chars() {
        let encoded = tokenizer.encode_ids(&letter.to_string())?;
        if encoded.len() != 1 {
            return Err(Error::Engine(format!(
                "Answer slot {letter:?} is not a shared single token"
            )));
        }
        let piece = model
            .vocab()
            .token_to_piece(LlamaToken(encoded[0]), true, None);
        let piece = String::from_utf8_lossy(&piece);
        if piece != letter.to_string() {
            return Err(Error::Engine(format!(
                "Answer slot {letter:?} is not a shared single token (piece={piece:?})"
            )));
        }
    }
    Ok(())
}

/// KV context size needed so parallel suffix decode does not hit NoKvCacheSlot.
/// `max_suffix_tokens` is the per-seq suffix headroom reserved beyond the shared prefix.
pub fn required_n_ctx(max_prompt_tokens: u32, n_seq_max: u32, max_suffix_tokens: u32) -> u32 {
    let seqs = n_seq_max.max(1);
    let suffix = max_suffix_tokens.max(1);
    max_prompt_tokens
        .saturating_add(seqs.saturating_mul(suffix))
        .saturating_add(64)
}

/// Deterministic head+tail truncation when a prompt exceeds `max_tokens`.
/// Keeps the first `head` and last `tail` tokens (head+tail == max_tokens when truncated).
pub fn truncate_tokens_head_tail(tokens: &[i32], max_tokens: usize) -> Vec<i32> {
    if tokens.len() <= max_tokens || max_tokens == 0 {
        return tokens.to_vec();
    }
    if max_tokens == 1 {
        return vec![tokens[0]];
    }
    let head = max_tokens / 2;
    let tail = max_tokens - head;
    let mut out = Vec::with_capacity(max_tokens);
    out.extend_from_slice(&tokens[..head]);
    out.extend_from_slice(&tokens[tokens.len() - tail..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_ctx_budget_covers_parallel_suffixes_vs_legacy() {
        let max_prompt = 2048u32;
        let n_seq = 8u32;
        let max_suffix = DEFAULT_MAX_SUFFIX_TOKENS;
        let need = required_n_ctx(max_prompt, n_seq, max_suffix);
        let legacy = max_prompt.saturating_add(64);
        assert_eq!(need, max_prompt + n_seq * max_suffix + 64);
        assert!(max_suffix > 64, "suffix headroom must exceed legacy 64");
        assert!(
            need > legacy,
            "legacy n_ctx={legacy} under-sizes parallel KV need={need} (NoKvCacheSlot risk)"
        );
    }

    #[test]
    fn truncate_tokens_head_tail_preserves_ends() {
        let toks: Vec<i32> = (0..20).collect();
        let out = truncate_tokens_head_tail(&toks, 10);
        assert_eq!(out.len(), 10);
        assert_eq!(&out[..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&out[5..], &[15, 16, 17, 18, 19]);
        assert_eq!(truncate_tokens_head_tail(&toks, 20), toks);
        assert_eq!(truncate_tokens_head_tail(&toks, 0), toks);
    }

    /// Two loads in one process used to fail with BackendAlreadyInitialized.
    /// Skips when ERESHKIGAL_GGUF is unset.
    #[test]
    fn shared_backend_allows_second_engine_load() {
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
        let gpu_layers: u32 = std::env::var("N_GPU_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let base = EngineConfig {
            gguf: path,
            tokenizer_source: tok_src,
            tokenizer_revision: tok_rev,
            max_prompt_tokens: 2048,
            threads: 4,
            n_gpu_layers: gpu_layers,
            n_seq_max: 1,
            embeddings: false,
            adapter: None,
            adapter_scale: 1.0,
        };
        let (a, _) = EngineOwned::load(base.clone()).expect("first engine load");
        let mut cpu = base;
        cpu.n_gpu_layers = 0;
        let (b, _) = EngineOwned::load(cpu).expect("second engine load (shared backend)");
        assert_eq!(a.meta.n_gpu_layers, gpu_layers as i32);
        assert_eq!(b.meta.n_gpu_layers, 0);
    }
}
