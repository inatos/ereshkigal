//! Bounded state outline for `state-outline-v1` (tree-sitter JSON + text fallbacks).
//! Does not alter `direct-options-v1` hashes.

use crate::error::{Error, Result};
use serde_json::Value;
use tree_sitter::{Parser, TreeCursor};

const MAX_CHARS: usize = 2048;
const MAX_NODES: usize = 64;

pub fn outline_state(state: &Value) -> Result<String> {
    match state {
        Value::String(s) => outline_text(s),
        other => {
            let dumped = serde_json::to_string(other)?;
            outline_json_source(&dumped)
        }
    }
}

fn outline_text(s: &str) -> Result<String> {
    let trimmed = s.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(out) = outline_json_source(trimmed) {
            return Ok(out);
        }
    }
    // Markdown / Python-ish: keep headings, defs, and first lines.
    let mut lines = Vec::new();
    for line in trimmed.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with('#')
            || t.starts_with("def ")
            || t.starts_with("class ")
            || t.starts_with("fn ")
            || t.starts_with("async ")
            || lines.len() < 8
        {
            lines.push(truncate(t, 160));
        }
        if lines.len() >= 24 {
            break;
        }
    }
    if lines.is_empty() {
        lines.push(truncate(trimmed, 400));
    }
    Ok(lines.join("\n"))
}

fn outline_json_source(src: &str) -> Result<String> {
    let mut parser = Parser::new();
    let lang = tree_sitter_json::LANGUAGE;
    parser
        .set_language(&lang.into())
        .map_err(|e| Error::Validation(format!("tree-sitter-json: {e}")))?;
    let tree = parser
        .parse(src, None)
        .ok_or_else(|| Error::Validation("tree-sitter failed to parse JSON state".into()))?;
    let mut out = Vec::new();
    let mut cursor = tree.walk();
    walk_json(&mut cursor, src, 0, &mut out);
    if out.is_empty() {
        out.push(truncate(src, 400));
    }
    let joined = out.join("\n");
    Ok(truncate(&joined, MAX_CHARS))
}

fn walk_json(cursor: &mut TreeCursor, src: &str, depth: usize, out: &mut Vec<String>) {
    if out.len() >= MAX_NODES || depth > 6 {
        return;
    }
    let node = cursor.node();
    let kind = node.kind();
    if matches!(kind, "pair" | "string" | "number" | "true" | "false" | "null") {
        let text = node.utf8_text(src.as_bytes()).unwrap_or("");
        if kind == "pair" || (kind != "string" && !text.is_empty()) {
            out.push(format!("{}{kind}: {}", indent(depth), truncate(text, 120)));
        }
    } else if kind == "object" || kind == "array" || kind == "document" {
        out.push(format!("{}{kind}", indent(depth)));
    }
    if cursor.goto_first_child() {
        loop {
            walk_json(cursor, src, depth + 1, out);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

fn indent(depth: usize) -> String {
    "  ".repeat(depth.min(8))
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max.saturating_sub(1)).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_object_outline_stable() {
        let a = outline_state(&json!({"health": "ok", "zones": 3})).unwrap();
        let b = outline_state(&json!({"health": "ok", "zones": 3})).unwrap();
        assert_eq!(a, b);
        assert!(a.contains("object") || a.contains("pair"));
    }

    #[test]
    fn whitespace_string_json_still_outlines() {
        let s = "  {\n  \"a\": 1\n}  ";
        let o = outline_state(&json!(s)).unwrap();
        assert!(!o.is_empty());
    }

    #[test]
    fn pythonish_defs() {
        let o = outline_state(&json!("def foo():\n    return 1\nclass Bar:\n    pass")).unwrap();
        assert!(o.contains("def foo"));
        assert!(o.contains("class Bar"));
    }
}
