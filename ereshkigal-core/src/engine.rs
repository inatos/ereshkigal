use crate::error::{Error, Result};
use crate::tokenizer::ReferenceTokenizer;
use crate::types::{GgufMeta, ModelMeta};
use llama_cpp_2::context::params::LlamaContextParams;
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

const DECODE_CHUNK: usize = 512;

pub struct EngineConfig {
    pub gguf: PathBuf,
    pub tokenizer_source: String,
    pub tokenizer_revision: String,
    pub max_prompt_tokens: usize,
    pub threads: i32,
    pub n_gpu_layers: u32,
}

/// Process-scoped llama.cpp scorer. The model is leaked for a `'static` context lifetime
/// (appropriate for a CLI or long-lived scoring process).
pub struct EngineOwned {
    _backend: LlamaBackend,
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    pub meta: ModelMeta,
    pub gguf_path: PathBuf,
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
        let backend = LlamaBackend::init().map_err(|e| Error::Engine(e.to_string()))?;
        let mut model_params = LlamaModelParams::default();
        if cfg.n_gpu_layers > 0 {
            model_params = model_params.with_n_gpu_layers(cfg.n_gpu_layers);
        }
        let model_params = pin!(model_params);
        let model = LlamaModel::load_from_file(&backend, &cfg.gguf, &model_params)
            .map_err(|e| Error::Engine(format!("load GGUF: {e}")))?;
        let model: &'static LlamaModel = Box::leak(Box::new(model));

        let ctx_tokens = (cfg.max_prompt_tokens as u32).saturating_add(64);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_tokens))
            .with_n_threads(cfg.threads)
            .with_n_threads_batch(cfg.threads)
            .with_n_seq_max(1);

        let ctx = model
            .new_context(&backend, ctx_params)
            .map_err(|e| Error::Engine(format!("create context: {e}")))?;

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
                _backend: backend,
                model,
                ctx,
                meta,
                gguf_path: cfg.gguf,
            },
            tokenizer,
        ))
    }

    pub fn clear(&mut self) {
        self.ctx.clear_kv_cache();
    }

    pub fn full_logits(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.clear();
        self.decode_logits(tokens, 0)
    }

    pub fn prefill(&mut self, prefix: &[i32]) -> Result<()> {
        let _ = self.decode_logits(prefix, 0)?;
        Ok(())
    }

    pub fn save_state(&self) -> Result<SeqState> {
        self.ctx
            .state_seq_get(0, LlamaStateSeqFlags::empty())
            .map_err(|e| Error::Engine(format!("state_seq_get: {e}")))
    }

    pub fn restore_state(&mut self, state: &SeqState) -> Result<()> {
        let ok = self
            .ctx
            .clear_kv_cache_seq(Some(0), None, None)
            .map_err(|e| Error::Engine(e.to_string()))?;
        if !ok {
            return Err(Error::Engine(
                "llama.cpp could not drop the previous scored branch".into(),
            ));
        }
        self.ctx
            .state_seq_set(state, 0)
            .map_err(|e| Error::Engine(format!("state_seq_set: {e}")))
    }

    pub fn branch_logits(&mut self, prefix_len: usize, suffix: &[i32]) -> Result<Vec<f32>> {
        self.decode_logits(suffix, prefix_len as i32)
    }

    pub fn gguf_tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let tokens = self.model.vocab().tokenize(text.as_bytes(), false, true);
        Ok(tokens.into_iter().map(|t| t.0).collect())
    }

    fn decode_logits(&mut self, tokens: &[i32], start: i32) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::Engine(
                "Refusing to decode an empty token list".into(),
            ));
        }
        let n_vocab = self.model.n_vocab() as usize;
        let mut batch = LlamaBatch::new(DECODE_CHUNK, 1);
        let total = tokens.len();
        for offset in (0..total).step_by(DECODE_CHUNK) {
            let end = (offset + DECODE_CHUNK).min(total);
            let chunk = &tokens[offset..end];
            batch.clear();
            for (i, &tok) in chunk.iter().enumerate() {
                let pos = start + offset as i32 + i as i32;
                let want_logits = offset + i == total - 1;
                batch
                    .add(LlamaToken(tok), pos, &[0], want_logits)
                    .map_err(|e| Error::Engine(format!("batch.add: {e}")))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| Error::Engine(format!(
                    "llama_decode failed; raise --max-tokens if prompts grew: {e}"
                )))?;
        }
        // llama-cpp-2 tracks batch offsets in initialized_logits; use contiguous last-token logits.
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
