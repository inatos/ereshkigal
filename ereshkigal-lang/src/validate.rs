use crate::error::{Error, Result};
use crate::types::{DecisionRow, LETTERS};
use serde_json::Value;
use std::collections::HashSet;

pub fn validate_row(row: &DecisionRow) -> Result<()> {
    if row.id.is_empty() {
        return Err(Error::Validation("id must be a nonempty string".into()));
    }
    if row.question.is_empty() {
        return Err(Error::Validation(
            "question must be a nonempty string".into(),
        ));
    }
    validate_state(&row.state)?;
    let n = row.options.len();
    if !(2..=LETTERS.len()).contains(&n) {
        return Err(Error::Validation(format!(
            "options must contain 2-{} entries",
            LETTERS.len()
        )));
    }
    let mut seen = HashSet::new();
    for option in &row.options {
        if option.id.is_empty() || option.description.is_empty() {
            return Err(Error::Validation(
                "Each option needs nonempty string id and description fields".into(),
            ));
        }
        if !seen.insert(option.id.as_str()) {
            return Err(Error::Validation("Option IDs must be unique".into()));
        }
    }
    Ok(())
}

fn validate_state(state: &Value) -> Result<()> {
    match state {
        Value::String(s) if !s.is_empty() => Ok(()),
        Value::Object(map) if !map.is_empty() => {
            serde_json::to_string(state).map_err(|e| Error::Validation(e.to_string()))?;
            Ok(())
        }
        Value::Array(arr) if !arr.is_empty() => {
            serde_json::to_string(state).map_err(|e| Error::Validation(e.to_string()))?;
            Ok(())
        }
        _ => Err(Error::Validation(
            "state must be a nonempty string, object, or array".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OptionSpec;
    use serde_json::json;

    fn ok_row() -> DecisionRow {
        DecisionRow {
            id: "r1".into(),
            state: json!("evidence"),
            question: "q?".into(),
            options: vec![
                OptionSpec {
                    id: "a".into(),
                    description: "A".into(),
                },
                OptionSpec {
                    id: "b".into(),
                    description: "B".into(),
                },
            ],
        }
    }

    #[test]
    fn accepts_minimal_row() {
        validate_row(&ok_row()).unwrap();
    }

    #[test]
    fn rejects_one_option() {
        let mut row = ok_row();
        row.options.pop();
        assert!(validate_row(&row).is_err());
    }

    #[test]
    fn rejects_duplicate_ids() {
        let mut row = ok_row();
        row.options[1].id = "a".into();
        assert!(validate_row(&row).is_err());
    }
}
