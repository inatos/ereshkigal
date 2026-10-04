//! Package manifest types (path / git / hf). Fetching lives in core.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum DepSource {
    Path { path: String },
    Git { git: String, rev: String },
    Hf { hf: String, rev: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Manifest {
    pub package: PackageMeta,
    #[serde(default)]
    pub dependencies: BTreeMap<String, DepSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PackageMeta {
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PackToml {
    pub name: String,
    #[serde(default)]
    pub base_gguf_sha256: String,
    #[serde(default)]
    pub tokenizer_revision: String,
    #[serde(default)]
    pub recipe: String,
    #[serde(default)]
    pub adapter: Option<String>,
    #[serde(default)]
    pub probe: Option<String>,
    #[serde(default)]
    pub training_data_sha256: Option<String>,
}

impl Manifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        toml::from_str(&s).map_err(|e| Error::Parse(e.to_string()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let s = toml::to_string_pretty(self).map_err(|e| Error::Parse(e.to_string()))?;
        std::fs::write(path, s)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_path_dep() {
        let s = r#"
[package]
name = "demo"
version = "0.1.0"
license = "MIT"

[dependencies]
std = { path = "decrees/std" }
"#;
        let m: Manifest = toml::from_str(s).unwrap();
        assert_eq!(m.package.name, "demo");
    }
}
