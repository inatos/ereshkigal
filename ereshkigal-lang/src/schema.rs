//! JSON Schema export for agent-written decrees.

use crate::decree::LibraryFile;
use crate::error::Result;
use schemars::schema_for;

pub fn library_schema_json() -> Result<String> {
    let schema = schema_for!(LibraryFile);
    Ok(serde_json::to_string_pretty(&schema)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_mentions_decrees() {
        let s = library_schema_json().unwrap();
        assert!(s.contains("decrees") || s.contains("LibraryFile"));
    }
}
