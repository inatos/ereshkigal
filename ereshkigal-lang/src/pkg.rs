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

/// Fingerprint a path dependency (file names + contents) for `fetch`/`verify`.
pub fn fingerprint_path(root: impl AsRef<Path>) -> Result<String> {
    use sha2::{Digest, Sha256};
    let root = root.as_ref();
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort();
    let mut h = Sha256::new();
    for rel in files {
        h.update(rel.as_bytes());
        h.update(b"\0");
        let abs = root.join(&rel);
        if let Ok(bytes) = std::fs::read(&abs) {
            h.update(&bytes);
        }
        h.update(b"\0");
    }
    Ok(hex::encode(h.finalize()))
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    if !dir.is_dir() {
        return Err(Error::Validation(format!("not a directory: {}", dir.display())));
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        let name = entry.file_name();
        if name == ".git" || name == "target" || name == "node_modules" {
            continue;
        }
        if p.is_dir() {
            collect_files(root, &p, out)?;
        } else if p.is_file() {
            let rel = p.strip_prefix(root).unwrap_or(&p);
            out.push(rel.to_string_lossy().into_owned());
        }
    }
    Ok(())
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
