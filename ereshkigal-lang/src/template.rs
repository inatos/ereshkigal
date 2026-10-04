//! Chat-template registry. Qwen3 thinking-off is the hash-pinned fast path.

use crate::error::{Error, Result};
use crate::prompt::render_qwen3_thinking_off;
use sha2::{Digest, Sha256};

pub const QWEN3_THINKING_OFF: &str = "qwen3_system_user_thinking_off";

pub fn render_chat(template_id: &str, messages: &[(String, String)]) -> Result<String> {
    match template_id {
        QWEN3_THINKING_OFF | "qwen3" | "default" => render_qwen3_thinking_off(messages),
        other => {
            // minijinja path is optional; unknown ids fail closed except a trivial passthrough.
            if other.starts_with("raw:") {
                Ok(messages
                    .iter()
                    .map(|(r, c)| format!("{r}: {c}"))
                    .collect::<Vec<_>>()
                    .join("\n"))
            } else {
                Err(Error::Validation(format!(
                    "unknown chat template {other}; use {QWEN3_THINKING_OFF} or embed via core minijinja"
                )))
            }
        }
    }
}

pub fn template_hash(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_hash_stable() {
        let m = vec![
            ("system".into(), "sys".into()),
            ("user".into(), "hi".into()),
        ];
        let a = render_chat(QWEN3_THINKING_OFF, &m).unwrap();
        let b = render_chat("default", &m).unwrap();
        assert_eq!(a, b);
        assert!(a.contains("<|im_start|>assistant"));
    }
}
