//! Confidence-gated AutoThink (Read-or-Think) deliberation.

use crate::decree::ThinkSpec;
use crate::engine::EngineOwned;
use crate::error::{Error, Result};
use crate::softmax::argmax;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    Auto,
    Never,
    Always,
}

impl ThinkMode {
    pub fn parse(s: &str) -> Self {
        match s {
            "always" => Self::Always,
            "never" => Self::Never,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThinkConfig {
    pub mode: ThinkMode,
    pub gate: f64,
    pub budget: usize,
    pub alpha: f64,
}

impl Default for ThinkConfig {
    fn default() -> Self {
        Self {
            mode: ThinkMode::Auto,
            gate: 0.90,
            budget: 512,
            alpha: 0.75,
        }
    }
}

impl ThinkConfig {
    pub fn from_spec(spec: &ThinkSpec) -> Self {
        Self {
            mode: ThinkMode::parse(&spec.mode),
            gate: spec.gate.clamp(0.0, 1.0),
            budget: spec.budget.min(512).max(1),
            alpha: spec.alpha.clamp(0.0, 1.0),
        }
    }

    /// Parse `/v1/systemone` or RPC `think` payload (bool, mode string, or object).
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        if v.is_null() {
            return None;
        }
        if let Some(b) = v.as_bool() {
            let mut cfg = Self::default();
            cfg.mode = if b { ThinkMode::Always } else { ThinkMode::Never };
            return Some(cfg);
        }
        if let Some(s) = v.as_str() {
            let mut cfg = Self::default();
            cfg.mode = ThinkMode::parse(s);
            return Some(cfg);
        }
        let obj = v.as_object()?;
        let mut cfg = Self::default();
        if let Some(m) = obj.get("mode").and_then(|x| x.as_str()) {
            cfg.mode = ThinkMode::parse(m);
        }
        if let Some(g) = obj.get("gate").and_then(|x| x.as_f64()) {
            cfg.gate = g.clamp(0.0, 1.0);
        }
        if let Some(b) = obj.get("budget").and_then(|x| x.as_u64()) {
            cfg.budget = (b as usize).min(512).max(1);
        }
        if let Some(a) = obj.get("alpha").and_then(|x| x.as_f64()) {
            cfg.alpha = a.clamp(0.0, 1.0);
        }
        Some(cfg)
    }
}

/// Richer AutoThink deliberation outcome returned by `Scorer::score_autothink`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutoThinkOutcome {
    pub probs: Vec<f64>,
    pub think_invoked: bool,
    pub think_tokens: usize,
    pub p0_vs_p1_flipped: bool,
}

/// Decide whether to deliberate given conformal set size and max p0.
pub fn should_think(cfg: &ThinkConfig, conformal_set_size: usize, max_p0: f64) -> bool {
    match cfg.mode {
        ThinkMode::Never => false,
        ThinkMode::Always => true,
        ThinkMode::Auto => conformal_set_size > 1 || max_p0 < cfg.gate,
    }
}

/// Fuse System-1 and System-2 distributions: `(1-α)·p0 + α·p1`.
pub fn fuse_probs(p0: &[f64], p1: &[f64], alpha: f64) -> Vec<f64> {
    let a = alpha.clamp(0.0, 1.0);
    let n = p0.len().min(p1.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((1.0 - a) * p0[i] + a * p1[i]);
    }
    let s: f64 = out.iter().sum::<f64>().max(1e-12);
    out.iter().map(|v| v / s).collect()
}

/// Apply temperature scaling in log-probability space (`softmax(ln(p) / T)`).
pub fn apply_temperature_to_probs(probs: &[f64], temperature: f64) -> Result<Vec<f64>> {
    let log_probs: Vec<f64> = probs.iter().map(|&p| p.max(1e-12).ln()).collect();
    crate::apply_temperature(&log_probs, temperature)
}

/// Fuse `p0` and `p1`, apply optional post-think temperature, and record whether
/// System-2 (`p1`) flipped the System-1 (`p0`) argmax.
pub fn build_autothink_outcome(
    p0: &[f64],
    p1: &[f64],
    think_tokens: usize,
    alpha: f64,
    post_think_t: Option<f64>,
) -> Result<AutoThinkOutcome> {
    let fused = fuse_probs(p0, p1, alpha);
    let probs = match post_think_t {
        Some(t) if t.is_finite() && t > 0.0 && (t - 1.0).abs() > 1e-12 => {
            apply_temperature_to_probs(&fused, t)?
        }
        _ => fused,
    };
    let a0 = argmax(p0);
    let a1 = argmax(p1);
    let p0_vs_p1_flipped = a0.is_some() && a1.is_some() && a0 != a1;
    Ok(AutoThinkOutcome {
        probs,
        think_invoked: true,
        think_tokens,
        p0_vs_p1_flipped,
    })
}

/// Whether a generated token should terminate thought generation (EOS, EOG, or `</think>`).
pub fn is_stop_token(tok: i32, eos: i32, is_eog: bool, think_end: Option<i32>) -> bool {
    is_eog || tok == eos || think_end == Some(tok)
}

/// Greedy thought generation (bounded). Prefers KV branch after prefix; falls
/// back to full decode of the growing sequence. Stops on EOS, `</think>`, loop, or budget.
pub fn generate_thought(
    engine: &mut EngineOwned,
    seed_tokens: &[i32],
    budget: usize,
) -> Result<(Vec<i32>, usize)> {
    if seed_tokens.is_empty() {
        return Err(Error::Engine("empty seed for thought".into()));
    }
    let budget = budget.min(512).max(1);
    let eos = engine.eos_token();
    let think_end = engine.think_end_token();
    let max_ctx = engine.meta.context_tokens as usize;
    let mut seq = seed_tokens.to_vec();
    let mut thought = Vec::new();
    let mut last_tok: Option<i32> = None;
    let mut repeat = 0usize;
    for _ in 0..budget {
        if seq.len() + 1 >= max_ctx {
            break;
        }
        let logits = if thought.is_empty() {
            engine.full_logits(&seq)?
        } else {
            let suf = [*thought.last().unwrap()];
            match engine.branch_logits(seq.len() - 1, &suf) {
                Ok(v) => v,
                Err(_) => engine.full_logits(&seq)?,
            }
        };
        let Some((tok, _)) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        else {
            break;
        };
        let tok = tok as i32;
        if is_stop_token(tok, eos, engine.is_eog_token(tok), think_end) {
            break;
        }
        if last_tok == Some(tok) {
            repeat += 1;
            if repeat >= 8 {
                break; // loop-break
            }
        } else {
            repeat = 0;
        }
        last_tok = Some(tok);
        thought.push(tok);
        seq.push(tok);
    }
    let n = thought.len();
    Ok((thought, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn auto_gate() {
        let cfg = ThinkConfig::default();
        assert!(!should_think(&cfg, 1, 0.95));
        assert!(should_think(&cfg, 2, 0.95));
        assert!(should_think(&cfg, 1, 0.5));
    }

    #[test]
    fn fuse_renorms() {
        let out = fuse_probs(&[0.8, 0.2], &[0.1, 0.9], 0.75);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn stop_token_triggers_on_eos_eog_and_think_end() {
        assert!(is_stop_token(151645, 151645, false, Some(151668)));
        assert!(is_stop_token(151668, 151645, false, Some(151668)));
        assert!(is_stop_token(999, 151645, true, Some(151668)));
        assert!(!is_stop_token(42, 151645, false, Some(151668)));
        assert!(!is_stop_token(42, 151645, false, None));
    }

    #[test]
    fn build_outcome_tracks_flip_and_post_think_temp() {
        let out = build_autothink_outcome(&[0.7, 0.3], &[0.1, 0.9], 42, 0.75, Some(2.0)).unwrap();
        assert!(out.think_invoked);
        assert_eq!(out.think_tokens, 42);
        assert!(out.p0_vs_p1_flipped);
        assert!((out.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(out.probs[1] > out.probs[0]);
        // Temperature > 1.0 softens fused distribution toward uniform.
        let unscaled = build_autothink_outcome(&[0.7, 0.3], &[0.1, 0.9], 42, 0.75, None).unwrap();
        assert!(out.probs[1] < unscaled.probs[1]);

        let no_flip =
            build_autothink_outcome(&[0.7, 0.3], &[0.85, 0.15], 12, 0.75, None).unwrap();
        assert!(!no_flip.p0_vs_p1_flipped);
    }

    #[test]
    fn think_config_parses_spec_and_json() {
        let spec = ThinkSpec {
            mode: "auto".into(),
            gate: 0.90,
            budget: 256,
            alpha: 0.75,
        };
        let cfg = ThinkConfig::from_spec(&spec);
        assert_eq!(cfg.mode, ThinkMode::Auto);
        assert_eq!(cfg.budget, 256);

        let j = ThinkConfig::from_json(&json!({"mode": "always", "budget": 64})).unwrap();
        assert_eq!(j.mode, ThinkMode::Always);
        assert_eq!(j.budget, 64);
        assert!((j.gate - 0.90).abs() < 1e-12);
        assert_eq!(ThinkConfig::from_json(&json!(false)).unwrap().mode, ThinkMode::Never);
        assert!(ThinkConfig::from_json(&serde_json::Value::Null).is_none());
    }
}
