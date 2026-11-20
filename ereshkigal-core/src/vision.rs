//! Optional mmproj / mtmd vision path (`--features vision`).
//!
//! Links `llama-cpp-2`'s `mtmd` multimodal projection bindings to decode image
//! evidence directly into token embeddings before reading last-position option
//! letter logits. Missing or unreadable `mmproj` weights always return `Err` so
//! callers cleanly abstain to exact text SoT.

use crate::engine::EngineOwned;
use crate::error::{Error, Result};
use crate::prompt::{digest, direct_messages, render_qwen3_thinking_off};
use crate::types::{DecisionRow, LETTERS};
use crate::validate::validate_row;
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use std::ffi::CString;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub mmproj: PathBuf,
    pub use_gpu: bool,
    pub n_threads: i32,
    pub image_min_tokens: i32,
    pub image_max_tokens: i32,
}

impl VisionConfig {
    pub fn from_mmproj(mmproj: impl Into<PathBuf>) -> Self {
        Self {
            mmproj: mmproj.into(),
            use_gpu: false,
            n_threads: 4,
            image_min_tokens: -1,
            image_max_tokens: -1,
        }
    }
}

/// Loaded multimodal projector (`mmproj`) bound to an `EngineOwned` text model.
pub struct VisionContext {
    ctx: MtmdContext,
    mmproj: PathBuf,
}

impl VisionContext {
    /// Load `mmproj` GGUF against `engine`'s underlying `LlamaModel`.
    /// Returns `Err` when `mmproj` is missing, unreadable, or lacks vision support
    /// so callers abstain cleanly to text.
    pub fn load(cfg: &VisionConfig, engine: &EngineOwned) -> Result<Self> {
        if !cfg.mmproj.is_file() {
            return Err(Error::Engine(format!(
                "mmproj not found: {}; abstain to text",
                cfg.mmproj.display()
            )));
        }
        let mmproj_str = cfg.mmproj.to_str().ok_or_else(|| {
            Error::Engine("mmproj path is not valid UTF-8; abstain to text".into())
        })?;
        let marker = CString::new(mtmd_default_marker())
            .map_err(|e| Error::Engine(format!("mtmd media marker CString: {e}")))?;
        let params = MtmdContextParams {
            use_gpu: cfg.use_gpu,
            print_timings: false,
            n_threads: cfg.n_threads.max(1),
            media_marker: marker,
            image_min_tokens: cfg.image_min_tokens,
            image_max_tokens: cfg.image_max_tokens,
        };
        let ctx = MtmdContext::init_from_file(mmproj_str, engine.llama_model(), &params)
            .map_err(|e| Error::Engine(format!("mtmd init_from_file failed: {e}; abstain to text")))?;
        if !ctx.support_vision() {
            return Err(Error::Engine(format!(
                "mmproj {} does not support vision; abstain to text",
                cfg.mmproj.display()
            )));
        }
        Ok(Self {
            ctx,
            mmproj: cfg.mmproj.clone(),
        })
    }

    pub fn support_vision(&self) -> bool {
        self.ctx.support_vision()
    }

    pub fn mmproj_path(&self) -> &Path {
        &self.mmproj
    }
}

/// Render a `direct-options-v1` chat prompt with exactly `n_images` copies of
/// `mtmd_default_marker()` prepended to the user message. Any accidental
/// literal occurrence of the marker inside evidence/question/options is
/// stripped first so `mtmd_tokenize` never trips `BitmapCountMismatch`.
pub fn render_visual_prompt(row: &DecisionRow, n_images: usize) -> Result<String> {
    if n_images == 0 {
        return Err(Error::Validation(
            "visual prompt requires at least one image".into(),
        ));
    }
    let marker = mtmd_default_marker();
    let mut messages = direct_messages(row)?;
    for (_, content) in &mut messages {
        if content.contains(marker) {
            *content = content.replace(marker, "");
        }
    }
    let mut prefix = String::with_capacity((marker.len() + 1) * n_images);
    for _ in 0..n_images {
        prefix.push_str(marker);
        prefix.push('\n');
    }
    messages[1].1 = format!("{prefix}{}", messages[1].1);
    render_qwen3_thinking_off(&messages)
}

/// Evaluate multimodal chunks (images + text prompt) into full vocabulary logits
/// at the final assistant readout token. Returns `(vocab_logits, total_tokens, prompt_sha256)`.
pub fn eval_visual_vocab_with_ctx(
    engine: &mut EngineOwned,
    vctx: &VisionContext,
    png_paths: &[&Path],
    row: &DecisionRow,
) -> Result<(Vec<f32>, usize, String)> {
    validate_row(row)?;
    if png_paths.is_empty() {
        return Err(Error::Validation(
            "visual scoring requires at least one image path; abstain to text".into(),
        ));
    }
    let mut bitmaps = Vec::with_capacity(png_paths.len());
    for path in png_paths {
        if !path.is_file() {
            return Err(Error::Engine(format!(
                "visual evidence image not found: {}; abstain to text",
                path.display()
            )));
        }
        let path_str = path.to_str().ok_or_else(|| {
            Error::Engine("visual evidence path is not valid UTF-8; abstain to text".into())
        })?;
        let bmp = MtmdBitmap::from_file(&vctx.ctx, path_str, false).map_err(|e| {
            Error::Engine(format!(
                "mtmd bitmap load failed for {}: {e}; abstain to text",
                path.display()
            ))
        })?;
        bitmaps.push(bmp);
    }

    let prompt = render_visual_prompt(row, bitmaps.len())?;
    let prompt_sha = digest(&prompt);
    let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
    let input = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };
    let chunks = vctx
        .ctx
        .tokenize(input, &bitmap_refs)
        .map_err(|e| Error::Engine(format!("mtmd tokenize failed: {e}; abstain to text")))?;
    let total_tokens = chunks.total_tokens();
    let max_ctx = engine.llama_context().n_ctx() as usize;
    if total_tokens == 0 || total_tokens > max_ctx {
        return Err(Error::Engine(format!(
            "mtmd total_tokens={total_tokens} exceeds context {max_ctx}; abstain to text"
        )));
    }

    engine.clear();
    let n_batch = engine.n_batch().max(1) as i32;
    chunks
        .eval_chunks(&vctx.ctx, engine.llama_context(), 0, 0, n_batch, true)
        .map_err(|e| Error::Engine(format!("mtmd eval_chunks failed: {e}; abstain to text")))?;

    let vocab = engine.last_vocab_logits()?;
    Ok((vocab, total_tokens, prompt_sha))
}

/// Extract option letter logits (`A..=P`) from a pre-loaded [`VisionContext`].
pub fn score_visual_logits_with_ctx(
    engine: &mut EngineOwned,
    vctx: &VisionContext,
    png_paths: &[&Path],
    row: &DecisionRow,
) -> Result<Vec<f32>> {
    let (vocab, _, _) = eval_visual_vocab_with_ctx(engine, vctx, png_paths, row)?;
    let mut out = Vec::with_capacity(row.options.len());
    for letter in LETTERS.chars().take(row.options.len()) {
        let toks = engine.gguf_tokenize(&letter.to_string())?;
        let tid = toks.last().copied().ok_or_else(|| {
            Error::Validation(format!("GGUF produced no token for letter {letter}"))
        })?;
        let logit = vocab
            .get(tid as usize)
            .copied()
            .ok_or_else(|| Error::Engine(format!("letter token id {tid} out of vocab range")))?;
        out.push(logit);
    }
    Ok(out)
}

/// One-shot visual letter-logit scoring conditioned on PNG evidence + `DecisionRow`.
///
/// Validates `row`, `cfg.mmproj`, and `png_paths` first so missing `mmproj`
/// weights or missing images always return `Err` for clean abstain to text SoT.
pub fn score_visual_logits(
    engine: Option<&mut EngineOwned>,
    cfg: &VisionConfig,
    png_paths: &[&Path],
    row: &DecisionRow,
) -> Result<Vec<f32>> {
    if !cfg.mmproj.is_file() {
        return Err(Error::Engine(format!(
            "mmproj not found: {}; abstain to text",
            cfg.mmproj.display()
        )));
    }
    validate_row(row)?;
    if png_paths.is_empty() {
        return Err(Error::Validation(
            "visual scoring requires at least one image path; abstain to text".into(),
        ));
    }
    for path in png_paths {
        if !path.is_file() {
            return Err(Error::Engine(format!(
                "visual evidence image not found: {}; abstain to text",
                path.display()
            )));
        }
    }
    let Some(engine) = engine else {
        return Err(Error::Engine(
            "no text GGUF engine loaded for mtmd vision; abstain to text".into(),
        ));
    };
    let vctx = VisionContext::load(cfg, engine)?;
    score_visual_logits_with_ctx(engine, &vctx, png_paths, row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionSpec;
    use serde_json::json;

    fn sample_row() -> DecisionRow {
        DecisionRow {
            id: "v".into(),
            state: json!("evidence"),
            question: "q?".into(),
            options: vec![
                OptionSpec {
                    id: "a".into(),
                    description: "Option A".into(),
                },
                OptionSpec {
                    id: "b".into(),
                    description: "Option B".into(),
                },
            ],
        }
    }

    #[test]
    fn missing_mmproj_abstains_with_err() {
        let cfg = VisionConfig::from_mmproj("/nonexistent/path/mmproj-missing.gguf");
        let row = sample_row();
        let err = score_visual_logits(None, &cfg, &[], &row).unwrap_err();
        assert!(err.to_string().contains("mmproj not found"), "{err}");
    }

    #[test]
    fn render_visual_prompt_matches_image_count_and_sanitizes_marker() {
        let marker = mtmd_default_marker();
        let mut row = sample_row();
        row.state = json!(format!("text containing literal {marker} marker"));
        let prompt = render_visual_prompt(&row, 2).unwrap();
        assert_eq!(prompt.matches(marker).count(), 2, "{prompt}");
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn render_visual_prompt_rejects_zero_images() {
        let row = sample_row();
        assert!(render_visual_prompt(&row, 0).is_err());
    }
}
