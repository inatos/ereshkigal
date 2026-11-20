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

/// Render `direct-options-v1` and return the exact byte offset of each option
/// object's closing `}` inside the rendered prompt string.
///
/// The returned prompt string is byte-for-byte identical to [`render_prompt`],
/// and the offsets are strictly increasing and immune to deceptive text inside
/// `evidence` or `criterion`. Used by cosine readout to align Rust and PyTorch
/// option hidden-state extraction positions.
pub fn render_direct_prompt_with_option_offsets(row: &DecisionRow) -> Result<(String, Vec<usize>)> {
    validate_row(row)?;
    let user_prefix = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
        DIRECT_SYSTEM.trim()
    );
    let mut user = String::new();
    user.push_str("{\"evidence\": ");
    user.push_str(&dumps_pythonish_inner(&row.state));
    user.push_str(", \"criterion\": ");
    user.push_str(&dumps_pythonish_inner(&Value::String(row.question.clone())));
    user.push_str(", \"options\": [");

    let mut option_end_offsets = Vec::with_capacity(row.options.len());
    for (index, option) in row.options.iter().enumerate() {
        if index > 0 {
            user.push_str(", ");
        }
        let letter = LETTERS
            .chars()
            .nth(index)
            .ok_or_else(|| Error::Validation("too many options".into()))?;
        user.push_str("{\"letter\": ");
        user.push_str(&dumps_pythonish_inner(&Value::String(letter.to_string())));
        user.push_str(", \"description\": ");
        user.push_str(&dumps_pythonish_inner(&Value::String(
            option.description.clone(),
        )));
        user.push('}');
        option_end_offsets.push(user_prefix.len() + user.len() - 1);
    }
    user.push_str("]}");

    let prompt = format!(
        "{user_prefix}{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    for &off in &option_end_offsets {
        if prompt.as_bytes().get(off) != Some(&b'}') {
            return Err(Error::Validation(
                "direct prompt option closing-brace offset mismatch".into(),
            ));
        }
    }
    Ok((prompt, option_end_offsets))
}

/// System prompt for multi-question packing (`joint-options-v1`).
const JOINT_SYSTEM: &str =
    "Score each FIELD by placing probability mass on OPTION letters at ANSWER:.";

fn parse_joint_field_options(raw: &[Value]) -> Option<Vec<crate::types::OptionSpec>> {
    if raw.len() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len());
    for (i, o) in raw.iter().enumerate() {
        if let Some(s) = o.as_str() {
            if s.trim().is_empty() {
                return None;
            }
            out.push(crate::types::OptionSpec {
                id: s.to_string(),
                description: s.to_string(),
            });
            continue;
        }
        let obj = o.as_object()?;
        let id = obj
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| format!("opt_{i}"));
        let description = obj
            .get("description")
            .or_else(|| obj.get("label"))
            .or_else(|| obj.get("desc"))
            .and_then(|v| v.as_str())
            .unwrap_or(id.as_str())
            .to_string();
        if id.trim().is_empty() || description.trim().is_empty() {
            return None;
        }
        out.push(crate::types::OptionSpec { id, description });
    }
    Some(out)
}

fn format_joint_user_with_colon_offsets(
    state: &Value,
    rows: &[DecisionRow],
) -> Result<(String, Vec<usize>)> {
    if rows.is_empty() {
        return Err(Error::Validation(
            "joint_options_batch_messages requires at least one row".into(),
        ));
    }
    for r in rows {
        validate_row(r)?;
    }
    let mut user = String::from("STATE:\n");
    match state {
        Value::String(s) => user.push_str(s),
        other => user.push_str(&dumps_pythonish_inner(other)),
    }
    user.push_str("\n\n");
    let mut colon_offsets = Vec::with_capacity(rows.len());
    for (q_idx, r) in rows.iter().enumerate() {
        if q_idx > 0 {
            user.push('\n');
        }
        user.push_str(&format!("FIELD {}:\n", q_idx + 1));
        user.push_str(&format!("QUESTION: {}\n", r.question));
        for (i, o) in r.options.iter().enumerate() {
            let letter = LETTERS.chars().nth(i).unwrap_or('?');
            user.push_str(&format!("OPTION {letter}: {}\n", o.description));
        }
        user.push_str("ANSWER:");
        colon_offsets.push(user.len() - 1);
        user.push('\n');
    }
    Ok((user, colon_offsets))
}

/// Multi-question packing (`joint-options-v1`): formats one shared `STATE`
/// followed by `FIELD 1..Q` (`QUESTION` / `OPTION A..` / `ANSWER:`).
pub fn joint_options_batch_messages(
    state: &Value,
    rows: &[DecisionRow],
) -> Result<Vec<(String, String)>> {
    let (user, _) = format_joint_user_with_colon_offsets(state, rows)?;
    Ok(vec![
        ("system".into(), JOINT_SYSTEM.to_string()),
        ("user".into(), user),
    ])
}

/// Render a `joint-options-v1` batch prompt and return the exact byte offset of
/// each `FIELD k`'s `ANSWER:` colon inside the rendered prompt string.
pub fn render_joint_batch_prompt_with_offsets(
    state: &Value,
    rows: &[DecisionRow],
) -> Result<(String, Vec<usize>)> {
    let (user, user_colons) = format_joint_user_with_colon_offsets(state, rows)?;
    let messages = vec![
        ("system".into(), JOINT_SYSTEM.to_string()),
        ("user".into(), user),
    ];
    let prompt = render_qwen3_thinking_off(&messages)?;
    let user_prefix = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
        JOINT_SYSTEM.trim()
    );
    let base = user_prefix.len();
    let colon_offsets: Vec<usize> = user_colons.into_iter().map(|o| base + o).collect();
    for &off in &colon_offsets {
        if prompt.as_bytes().get(off) != Some(&b':') {
            return Err(Error::Validation(
                "joint prompt ANSWER: colon offset mismatch".into(),
            ));
        }
    }
    Ok((prompt, colon_offsets))
}

/// Multi-question packing (`joint-options-v1`): STATE + FIELD blocks + ANSWER: markers.
/// `row.state` may be a JSON array of `{question, options}` fields; single-row falls back.
pub fn joint_options_messages(row: &DecisionRow) -> Result<Vec<(String, String)>> {
    validate_row(row)?;
    if let Value::Array(items) = &row.state {
        if !items.is_empty() {
            let mut fields = Vec::with_capacity(items.len());
            let mut all_fields = true;
            let shared_state = items
                .first()
                .and_then(|v| v.get("state").or_else(|| v.get("evidence")))
                .cloned()
                .unwrap_or_else(|| Value::String(row.question.clone()));
            for (idx, item) in items.iter().enumerate() {
                let Some(obj) = item.as_object() else {
                    all_fields = false;
                    break;
                };
                let Some(q) = obj.get("question").and_then(|v| v.as_str()) else {
                    all_fields = false;
                    break;
                };
                let Some(raw_opts) = obj.get("options").and_then(|v| v.as_array()) else {
                    all_fields = false;
                    break;
                };
                let Some(options) = parse_joint_field_options(raw_opts) else {
                    all_fields = false;
                    break;
                };
                let fid = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{}-{}", row.id, idx + 1));
                fields.push(DecisionRow {
                    id: fid,
                    state: shared_state.clone(),
                    question: q.to_string(),
                    options,
                });
            }
            if all_fields && !fields.is_empty() {
                return joint_options_batch_messages(&shared_state, &fields);
            }
        }
    }
    joint_options_batch_messages(&row.state, std::slice::from_ref(row))
}

pub fn render_prompt_version(row: &DecisionRow, version: &str) -> Result<String> {
    let messages = match version {
        PROMPT_VERSION => direct_messages(row)?,
        crate::types::PROMPT_VERSION_JOINT => joint_options_messages(row)?,
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

    #[test]
    fn joint_batch_keeps_direct_sha_and_emits_answer_markers() {
        let row1 = support_row();
        let v1_sha = digest(&render_prompt(&row1).unwrap());
        assert_eq!(
            v1_sha,
            "7ac35785358f0c656eeb741ca0a51f03e0133753a71e2879a18ad0ab56d0c024"
        );

        let row2 = DecisionRow {
            id: "support-2".into(),
            state: row1.state.clone(),
            question: "Was any rollback initiated?".into(),
            options: vec![
                OptionSpec {
                    id: "yes".into(),
                    description: "A rollback was initiated.".into(),
                },
                OptionSpec {
                    id: "no".into(),
                    description: "No rollback was initiated.".into(),
                },
            ],
        };
        let row3 = DecisionRow {
            id: "support-3".into(),
            state: row1.state.clone(),
            question: "Did health checks pass?".into(),
            options: vec![
                OptionSpec {
                    id: "all".into(),
                    description: "All zones passed.".into(),
                },
                OptionSpec {
                    id: "partial".into(),
                    description: "Only some zones passed.".into(),
                },
            ],
        };

        let msgs = joint_options_batch_messages(&row1.state, &[row1.clone(), row2, row3]).unwrap();
        assert_eq!(msgs.len(), 2);
        let user = &msgs[1].1;
        assert!(user.contains("FIELD 1:\n"));
        assert!(user.contains("FIELD 2:\n"));
        assert!(user.contains("FIELD 3:\n"));
        assert_eq!(user.matches("ANSWER:\n").count(), 3);

        // Verify exact colon byte offsets even when STATE contains deceptive ANSWER: text.
        let tricky_state = json!("Logs say FIELD 1: ANSWER: fake.");
        let mut r_a = row1.clone();
        r_a.state = tricky_state.clone();
        let mut r_b = row1.clone();
        r_b.id = "support-b".into();
        r_b.state = tricky_state.clone();
        let (prompt, colons) =
            render_joint_batch_prompt_with_offsets(&tricky_state, &[r_a, r_b]).unwrap();
        assert_eq!(colons.len(), 2);
        for &c in &colons {
            assert_eq!(prompt.as_bytes()[c], b':');
            assert!(prompt[..=c].ends_with("ANSWER:"));
        }

        // Verify joint_options_messages expands JSON array of {question, options}.
        let array_row = DecisionRow {
            id: "joint-arr".into(),
            state: json!([
                {
                    "state": "Shared evidence body.",
                    "question": "First question?",
                    "options": [
                        {"id": "a", "description": "Alpha"},
                        {"id": "b", "description": "Beta"}
                    ]
                },
                {
                    "question": "Second question?",
                    "options": ["Yes", "No"]
                }
            ]),
            question: "Fallback header".into(),
            options: row1.options.clone(),
        };
        let arr_msgs = joint_options_messages(&array_row).unwrap();
        assert!(arr_msgs[1].1.contains("STATE:\nShared evidence body."));
        assert!(arr_msgs[1].1.contains("FIELD 1:\nQUESTION: First question?"));
        assert!(arr_msgs[1].1.contains("FIELD 2:\nQUESTION: Second question?"));
        assert_eq!(arr_msgs[1].1.matches("ANSWER:\n").count(), 2);

        // Direct v1 SHA remains untouched after joint calls.
        assert_eq!(
            digest(&render_prompt(&row1).unwrap()),
            "7ac35785358f0c656eeb741ca0a51f03e0133753a71e2879a18ad0ab56d0c024"
        );
    }

    #[test]
    fn direct_option_offsets_match_render_prompt_bytes() {
        let mut row = support_row();
        row.state = json!("Fake {\"letter\": \"A\", \"description\": \"decoy\"} inside evidence.");
        let plain = render_prompt(&row).unwrap();
        let (with_off, offsets) = render_direct_prompt_with_option_offsets(&row).unwrap();
        assert_eq!(plain, with_off);
        assert_eq!(offsets.len(), row.options.len());
        for w in offsets.windows(2) {
            assert!(w[0] < w[1]);
        }
        for (i, &off) in offsets.iter().enumerate() {
            assert_eq!(with_off.as_bytes()[off], b'}');
            assert!(with_off[..=off].contains(&row.options[i].description));
        }
    }
}
