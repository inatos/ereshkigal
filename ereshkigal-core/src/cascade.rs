//! Letter-slot cascade: draft model commits when margin is large, else verify.
//!
//! Leviathan-style speculative decoding adapted to option logits (no generated
//! answer tokens). Argmax of the committed distribution is the decision.

use crate::error::{Error, Result};
use crate::softmax::{argmax, softmax};

#[derive(Debug, Clone, PartialEq)]
pub struct CascadeOutcome {
    pub probabilities: Vec<f64>,
    pub used_verify: bool,
    pub draft_margin: f64,
    pub source: &'static str,
}

/// Top-1 minus top-2 probability from logits.
pub fn top2_margin(logits: &[f64]) -> Result<f64> {
    let p = softmax(logits)?;
    let mut s = p.clone();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(s[0] - s[1])
}

/// If draft margin > `tau`, commit draft softmax; else use `verify_logits`.
pub fn cascade_select(
    draft_logits: &[f64],
    verify_logits: Option<&[f64]>,
    tau: f64,
) -> Result<CascadeOutcome> {
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(Error::Validation("cascade tau must be finite and >= 0".into()));
    }
    let margin = top2_margin(draft_logits)?;
    if margin > tau {
        return Ok(CascadeOutcome {
            probabilities: softmax(draft_logits)?,
            used_verify: false,
            draft_margin: margin,
            source: "cascade-draft",
        });
    }
    let verify = verify_logits.ok_or_else(|| {
        Error::Validation("cascade requires verify logits when margin <= tau".into())
    })?;
    if verify.len() != draft_logits.len() {
        return Err(Error::Validation(
            "draft and verify option counts differ".into(),
        ));
    }
    Ok(CascadeOutcome {
        probabilities: softmax(verify)?,
        used_verify: true,
        draft_margin: margin,
        source: "cascade-verify",
    })
}

/// Whether a draft-only commit would keep the same argmax as verify (diagnostic).
pub fn draft_agrees_argmax(draft_logits: &[f64], verify_logits: &[f64]) -> Result<bool> {
    let d = softmax(draft_logits)?;
    let v = softmax(verify_logits)?;
    Ok(argmax(&d) == argmax(&v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_margin_skips_verify() {
        let draft = vec![8.0, 0.0, 0.1];
        let verify = vec![0.0, 8.0, 0.0]; // would flip if used
        let out = cascade_select(&draft, Some(&verify), 0.2).unwrap();
        assert!(!out.used_verify);
        assert_eq!(argmax(&out.probabilities), Some(0));
        assert!(out.draft_margin > 0.2);
    }

    #[test]
    fn low_margin_uses_verify() {
        let draft = vec![1.0, 0.9, 0.8];
        let verify = vec![0.0, 3.0, 0.1];
        let out = cascade_select(&draft, Some(&verify), 0.5).unwrap();
        assert!(out.used_verify);
        assert_eq!(argmax(&out.probabilities), Some(1));
    }
}
