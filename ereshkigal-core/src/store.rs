//! Persistent prompt-hash replay store (JSONL).

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoreRow {
    key: String,
    logits: Vec<f64>,
    option_ids: Vec<String>,
    prompt_sha256: String,
}

#[derive(Debug, Default)]
pub struct ReplayStore {
    path: Option<PathBuf>,
    inner: HashMap<String, StoreRow>,
}

impl ReplayStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut inner = HashMap::new();
        if path.is_file() {
            let f = File::open(&path)?;
            for line in BufReader::new(f).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let row: StoreRow = serde_json::from_str(&line)?;
                inner.insert(row.key.clone(), row);
            }
        }
        Ok(Self {
            path: Some(path),
            inner,
        })
    }

    pub fn key(
        prompt_sha: &str,
        gguf_sha: &str,
        adapter_sha: Option<&str>,
        template_id: &str,
        readout: &str,
    ) -> String {
        format!(
            "{prompt_sha}|{gguf_sha}|{}|{template_id}|{readout}",
            adapter_sha.unwrap_or("-")
        )
    }

    pub fn get(&self, key: &str) -> Option<&[f64]> {
        self.inner.get(key).map(|r| r.logits.as_slice())
    }

    pub fn insert(&mut self, key: String, logits: Vec<f64>, option_ids: Vec<String>, prompt_sha256: String) -> Result<()> {
        let row = StoreRow {
            key: key.clone(),
            logits,
            option_ids,
            prompt_sha256,
        };
        if let Some(path) = &self.path {
            let mut f = OpenOptions::new().create(true).append(true).open(path)?;
            serde_json::to_writer(&mut f, &row)?;
            f.write_all(b"\n")?;
        }
        self.inner.insert(key, row);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl ReplayStore {
    pub fn memory() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_roundtrip() {
        let mut s = ReplayStore::memory();
        s.insert("k".into(), vec![1.0, 2.0], vec!["a".into(), "b".into()], "p".into())
            .unwrap();
        assert_eq!(s.get("k").unwrap()[1], 2.0);
    }
}
