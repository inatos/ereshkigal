use crate::error::{Error, Result};
use crate::types::{DecisionRow, EncodedDecision, DIRECT_SYSTEM, LETTERS, PROMPT_VERSION};
use crate::validate::validate_row;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Match Python `json.dumps(obj, ensure_ascii=False)` default separators (", ", ": ").
fn dumps_pythonish(value: &Value) -> Result<String> {
    Ok(dumps_pythonish_inner(value))
}

fn dumps_pythonish_inner(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(&Value::String(s.clone())).unwrap(),
        Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(dumps_pythonish_inner).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    let key = serde_json::to_string(&Value::String(k.clone())).unwrap();
                    format!("{}: {}", key, dumps_pythonish_inner(v))
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}


/// Render SemIf direct messages (system + user JSON payload).
pub fn direct_messages(row: &DecisionRow) -> Result<Vec<(String, String)>> {
    validate_row(row)?;
    let mut options = Vec::with_capacity(row.options.len());
    for (index, option) in row.options.iter().enumerate() {
        let mut opt = serde_json::Map::new();
        opt.insert(
            "letter".into(),
            Value::String(LETTERS.chars().nth(index).unwrap().to_string()),
        );
        opt.insert(
            "description".into(),
            Value::String(option.description.clone()),
        );
        options.push(Value::Object(opt));
    }
    let mut payload = serde_json::Map::new();
    payload.insert("evidence".into(), row.state.clone());
    payload.insert("criterion".into(), Value::String(row.question.clone()));
    payload.insert("options".into(), Value::Array(options));
    let user = dumps_pythonish(&Value::Object(payload))?;
    Ok(vec![
        ("system".into(), DIRECT_SYSTEM.to_string()),
        ("user".into(), user),
    ])
}

fn option_array(row: &DecisionRow) -> Vec<Value> {
    let mut options = Vec::with_capacity(row.options.len());
    for (index, option) in row.options.iter().enumerate() {
        let mut opt = serde_json::Map::new();
        opt.insert(
            "letter".into(),
            Value::String(LETTERS.chars().nth(index).unwrap().to_string()),
        );
        opt.insert(
            "description".into(),
            Value::String(option.description.clone()),
        );
        options.push(Value::Object(opt));
    }
    options
}

/// Criterion + options before evidence so a shared prefix covers many items.
pub fn options_first_messages(row: &DecisionRow) -> Result<Vec<(String, String)>> {
    validate_row(row)?;
    let mut payload = serde_json::Map::new();
    payload.insert("criterion".into(), Value::String(row.question.clone()));
    payload.insert("options".into(), Value::Array(option_array(row)));
    payload.insert("evidence".into(), row.state.clone());
    let user = dumps_pythonish(&Value::Object(payload))?;
    Ok(vec![
        ("system".into(), DIRECT_SYSTEM.to_string()),
        ("user".into(), user),
    ])
}

fn probe_state_messages(row: &DecisionRow) -> Result<Vec<(String, String)>> {
    validate_row(row)?;
    let mut payload = serde_json::Map::new();
    payload.insert("evidence".into(), row.state.clone());
    let user = dumps_pythonish(&Value::Object(payload))?;
    Ok(vec![
        (
            "system".into(),
            "Encode the supplied evidence. Do not answer.".into(),
        ),
        ("user".into(), user),
    ])
}

fn simple_outline(state: &Value) -> String {
    match state {
        Value::String(s) => s.chars().take(2048).collect(),
        other => {
            let dumped = serde_json::to_string(other).unwrap_or_default();
            dumped.chars().take(2048).collect()
        }
    }
}

/// Qwen3 chat template for system+user with `enable_thinking=false`.
///
/// Matches transformers `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`
/// for the SemIf message shape (verified against Qwen/Qwen3.5-4B @ 851bf6e8).
pub fn render_qwen3_thinking_off(messages: &[(String, String)]) -> Result<String> {
    if messages.len() != 2 || messages[0].0 != "system" || messages[1].0 != "user" {
        return Err(Error::Validation(
            "qwen3_system_user_thinking_off expects [system, user]".into(),
        ));
    }
    let system = messages[0].1.trim();
    let user = messages[1].1.trim();
    Ok(format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ))
}

pub fn render_prompt(row: &DecisionRow) -> Result<String> {
    render_prompt_version(row, PROMPT_VERSION)
}

pub fn render_prompt_version(row: &DecisionRow, version: &str) -> Result<String> {
    let messages = match version {
        PROMPT_VERSION => direct_messages(row)?,
        crate::types::PROMPT_VERSION_OUTLINE => {
            let outline = simple_outline(&row.state);
            let mut clone = row.clone();
            clone.state = Value::String(format!("[state-outline-v1]\n{outline}"));
            direct_messages(&clone)?
        }
        crate::types::PROMPT_VERSION_OPTIONS_FIRST => options_first_messages(row)?,
        crate::types::PROMPT_VERSION_PROBE_DECREE => direct_messages(row)?,
        crate::types::PROMPT_VERSION_PROBE_STATE => probe_state_messages(row)?,
        other => {
            return Err(Error::Validation(format!("unknown prompt_version {other}")));
        }
    };
    render_qwen3_thinking_off(&messages)
}

pub fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn prompt_version() -> &'static str {
    PROMPT_VERSION
}

/// Evidence-prefix token boundary used by serial/shared modes (upstream `_state_prefix`).
pub fn state_prefix_text(row: &DecisionRow) -> Result<String> {
    state_prefix_text_version(row, PROMPT_VERSION)
}

pub fn state_prefix_text_version(row: &DecisionRow, version: &str) -> Result<String> {
    let prompt = render_prompt_version(row, version)?;
    let messages = match version {
        crate::types::PROMPT_VERSION_OUTLINE => {
            let outline = simple_outline(&row.state);
            let mut clone = row.clone();
            clone.state = Value::String(format!("[state-outline-v1]\n{outline}"));
            direct_messages(&clone)?
        }
        crate::types::PROMPT_VERSION_OPTIONS_FIRST => options_first_messages(row)?,
        crate::types::PROMPT_VERSION_PROBE_STATE => probe_state_messages(row)?,
        _ => direct_messages(row)?,
    };
    let payload = &messages[1].1;
    let idx = prompt
        .find(payload.as_str())
        .ok_or_else(|| Error::msg("payload missing from prompt"))?;
    if version == crate::types::PROMPT_VERSION_OPTIONS_FIRST {
        let needle = ", \"evidence\": ";
        let rel = payload
            .find(needle)
            .ok_or_else(|| Error::Validation("options-first payload missing evidence key".into()))?;
        // Include up to and including `"evidence": ` so items share criterion+options.
        let cut = rel + needle.len();
        return Ok(format!("{}{}", &prompt[..idx], &payload[..cut]));
    }
    let mut evidence_obj = serde_json::Map::new();
    evidence_obj.insert("evidence".into(), row.state.clone());
    let evidence = dumps_pythonish(&Value::Object(evidence_obj))?;
    let evidence = &evidence[..evidence.len() - 1]; // drop trailing `}`
    let count = prompt.matches(payload.as_str()).count();
    if count != 1 || !payload.starts_with(evidence) {
        return Err(Error::Validation(
            "Cannot establish a deterministic evidence prefix".into(),
        ));
    }
    Ok(format!("{}{}", &prompt[..idx], evidence))
}

pub fn encode_with_ids(
    prompt: String,
    token_ids: Vec<i32>,
    slots: Vec<i32>,
) -> EncodedDecision {
    let prompt_sha256 = digest(&prompt);
    EncodedDecision {
        token_ids,
        slots,
        prompt,
        prompt_sha256,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionSpec;
    use serde_json::json;

    fn support_row() -> DecisionRow {
        DecisionRow {
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
        }
    }

    #[test]
    fn support1_prompt_sha_matches_transformers() {
        let prompt = render_prompt(&support_row()).unwrap();
        assert_eq!(
            digest(&prompt),
            "7ac35785358f0c656eeb741ca0a51f03e0133753a71e2879a18ad0ab56d0c024"
        );
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn route1_and_policy1_hashes() {
        let route = DecisionRow {
            id: "route-1".into(),
            state: json!(
                "Customer asks to reset a forgotten password and says the reset email never arrived."
            ),
            question: "Which queue should handle this request?".into(),
            options: vec![
                OptionSpec {
                    id: "account_access".into(),
                    description: "Account access and authentication support.".into(),
                },
                OptionSpec {
                    id: "billing".into(),
                    description: "Billing and payment support.".into(),
                },
                OptionSpec {
                    id: "sales".into(),
                    description: "Sales and product evaluation.".into(),
                },
            ],
        };
        assert_eq!(
            digest(&render_prompt(&route).unwrap()),
            "016267542103ab91422c425d2234ec7e1b9aef77f9d3e481b96db2de7cca470a"
        );

        let policy = DecisionRow {
            id: "policy-1".into(),
            state: json!(
                "Policy: production deletion requires an approved change ticket. Request: list the names of files in the production backup; do not modify anything."
            ),
            question: "Does the request require an approved change ticket under the stated policy?"
                .into(),
            options: vec![
                OptionSpec {
                    id: "required".into(),
                    description: "An approved change ticket is required.".into(),
                },
                OptionSpec {
                    id: "not_required".into(),
                    description: "An approved change ticket is not required.".into(),
                },
                OptionSpec {
                    id: "insufficient".into(),
                    description: "The evidence is insufficient to decide.".into(),
                },
            ],
        };
        assert_eq!(
            digest(&render_prompt(&policy).unwrap()),
            "6884fc4a9117b63369dd79e3042b92f92678f1e08dba2600e93d1da6754b1787"
        );
    }

    #[test]
    fn outline_version_does_not_change_v1_hash() {
        let v1 = digest(&render_prompt(&support_row()).unwrap());
        assert_eq!(
            v1,
            "7ac35785358f0c656eeb741ca0a51f03e0133753a71e2879a18ad0ab56d0c024"
        );
        let outline = render_prompt_version(&support_row(), crate::types::PROMPT_VERSION_OUTLINE)
            .unwrap();
        assert_ne!(digest(&outline), v1);
        assert!(outline.contains("[state-outline-v1]"));
    }

    #[test]
    fn options_first_differs_and_prefixes_before_evidence() {
        let v1 = digest(&render_prompt(&support_row()).unwrap());
        let first = render_prompt_version(
            &support_row(),
            crate::types::PROMPT_VERSION_OPTIONS_FIRST,
        )
        .unwrap();
        assert_ne!(digest(&first), v1);
        let prefix =
            state_prefix_text_version(&support_row(), crate::types::PROMPT_VERSION_OPTIONS_FIRST)
                .unwrap();
        assert!(prefix.contains("\"criterion\""));
        assert!(prefix.contains("\"options\""));
        assert!(prefix.ends_with("\"evidence\": ") || prefix.contains("\"evidence\": "));
        assert!(!prefix.contains("14:02 UTC"));
    }
}
