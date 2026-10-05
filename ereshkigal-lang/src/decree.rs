//! Shared decree IR (loaded from TOML, JSON, or .esk).

use crate::error::{Error, Result};
use crate::types::{OptionSpec, LETTERS};
use crate::validate::validate_row;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DecreeKind {
    #[default]
    Enum,
    Bool,
    Ordinal,
    Multilabel,
    Tree,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OnAbstain {
    #[default]
    Return,
    Escalate,
    Fail,
    Default,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DebiasMode {
    #[default]
    None,
    ContentFree,
    Permute,
    /// Zheng et al. ICLR 2024 PriDe: letter prior from cyclic subset, then p/prior.
    Pride,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    #[default]
    Letter,
    Probe,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct AbstainSpec {
    #[serde(default)]
    pub coverage: Option<f64>,
    #[serde(default)]
    pub min_margin: Option<f64>,
    #[serde(default)]
    pub min_probability: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct GoldTest {
    pub state: Value,
    pub expect: String,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub split: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
pub struct Variant {
    pub id: String,
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub options: Option<Vec<OptionSpec>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
pub struct TreeGroup {
    pub id: String,
    pub description: String,
    pub children: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Decree {
    pub name: String,
    pub question: String,
    pub options: Vec<OptionSpec>,
    #[serde(default)]
    pub kind: DecreeKind,
    #[serde(default)]
    pub abstain: Option<AbstainSpec>,
    #[serde(default)]
    pub on_abstain: OnAbstain,
    #[serde(default)]
    pub default_id: Option<String>,
    #[serde(default)]
    pub costs: BTreeMap<String, BTreeMap<String, f64>>,
    #[serde(default)]
    pub tests: Vec<GoldTest>,
    #[serde(default)]
    pub variants: Vec<Variant>,
    #[serde(default)]
    pub backend: BackendKind,
    #[serde(default)]
    pub debias: DebiasMode,
    #[serde(default)]
    pub tree: Option<Vec<TreeGroup>>,
    #[serde(default)]
    pub recipe: Option<String>,
}

impl Decree {
    pub fn option_index(&self, id: &str) -> Option<usize> {
        self.options.iter().position(|o| o.id == id)
    }

    pub fn to_row(&self, id: impl Into<String>, state: Value) -> Result<crate::types::DecisionRow> {
        if self.kind == DecreeKind::Tree {
            return Err(Error::Validation(
                "tree decrees must be lowered into ≤16-way rows".into(),
            ));
        }
        let row = crate::types::DecisionRow {
            id: id.into(),
            state,
            question: self.question.clone(),
            options: self.options.clone(),
        };
        validate_row(&row)?;
        if row.options.len() > LETTERS.len() {
            return Err(Error::Validation("more than 16 options requires kind=tree".into()));
        }
        Ok(row)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
pub enum Guard {
    Is {
        node: String,
        #[serde(rename = "is")]
        is: String,
    },
    In {
        node: String,
        #[serde(rename = "in")]
        r#in: Vec<String>,
    },
    PAtLeast {
        node: String,
        p_at_least: f64,
        #[serde(default)]
        id: Option<String>,
    },
    All {
        all: Vec<Guard>,
    },
    Any {
        any: Vec<Guard>,
    },
    Not {
        not: Box<Guard>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ForEachOp {
    #[default]
    Map,
    Filter,
    TopK,
    SortPairwise,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ForEach {
    pub path: String,
    pub op: ForEachOp,
    #[serde(default)]
    pub k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(default)]
pub struct ProgramNode {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub decree: String,
    #[serde(default)]
    pub when: Option<Guard>,
    #[serde(default)]
    pub with: Vec<String>,
    #[serde(default)]
    pub foreach: Option<ForEach>,
    #[serde(default)]
    pub pairwise: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
pub struct MatchArm {
    pub when: Vec<String>,
    pub then: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(default)]
pub struct Program {
    pub name: String,
    #[serde(default)]
    pub nodes: Vec<ProgramNode>,
    #[serde(default)]
    pub result: Vec<MatchArm>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
pub struct LibraryFile {
    #[serde(default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub recipe: Option<String>,
    #[serde(default)]
    pub decrees: BTreeMap<String, DecreeFileEntry>,
    #[serde(default)]
    pub programs: BTreeMap<String, Program>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
pub struct DecreeFileEntry {
    pub question: String,
    pub options: Vec<OptionSpec>,
    #[serde(default)]
    pub kind: DecreeKind,
    #[serde(default)]
    pub abstain: Option<AbstainSpec>,
    #[serde(default)]
    pub on_abstain: OnAbstain,
    #[serde(default)]
    pub default_id: Option<String>,
    #[serde(default)]
    pub costs: BTreeMap<String, BTreeMap<String, f64>>,
    #[serde(default)]
    pub tests: Vec<GoldTest>,
    #[serde(default)]
    pub variants: Vec<Variant>,
    #[serde(default)]
    pub backend: BackendKind,
    #[serde(default)]
    pub debias: DebiasMode,
    #[serde(default)]
    pub tree: Option<Vec<TreeGroup>>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Library {
    pub recipe: String,
    pub decrees: BTreeMap<String, Decree>,
    pub programs: BTreeMap<String, Program>,
    pub sources: Vec<String>,
}

impl Library {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut lib = Library {
            recipe: crate::types::PROMPT_VERSION.to_string(),
            ..Default::default()
        };
        lib.load_into(path.as_ref())?;
        crate::lint::lint_library(&lib)?;
        Ok(lib)
    }

    /// Parse a single TOML library document from a string (LSP / serve helpers).
    pub fn load_from_toml_str(src: &str) -> Result<Self> {
        let mut lib = Library {
            recipe: crate::types::PROMPT_VERSION.to_string(),
            ..Default::default()
        };
        let file: LibraryFile =
            toml::from_str(src).map_err(|e| Error::Parse(e.to_string()))?;
        lib.merge_file(file)?;
        Ok(lib)
    }

    pub fn decree(&self, name: &str) -> Result<&Decree> {
        self.decrees
            .get(name)
            .ok_or_else(|| Error::Validation(format!("unknown decree {name}")))
    }

    pub fn program(&self, name: &str) -> Result<&Program> {
        self.programs
            .get(name)
            .ok_or_else(|| Error::Validation(format!("unknown program {name}")))
    }

    fn load_into(&mut self, path: &Path) -> Result<()> {
        if path.is_dir() {
            let mut ents: Vec<_> = std::fs::read_dir(path)?.filter_map(|e| e.ok()).collect();
            ents.sort_by_key(|e| e.path());
            for e in ents {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("toml")
                    || p.extension().and_then(|s| s.to_str()) == Some("json")
                    || p.extension().and_then(|s| s.to_str()) == Some("esk")
                {
                    self.load_into(&p)?;
                } else if p.is_dir() {
                    self.load_into(&p)?;
                }
            }
            return Ok(());
        }
        let src = std::fs::read_to_string(path)?;
        self.sources.push(path.display().to_string());
        match path.extension().and_then(|s| s.to_str()) {
            Some("esk") => {
                let parsed = crate::syntax::parse_library(&src)?;
                self.merge(parsed)?;
            }
            Some("json") => {
                let file: LibraryFile = serde_json::from_str(&src)?;
                self.merge_file(file)?;
            }
            _ => {
                let file: LibraryFile = toml::from_str(&src)
                    .map_err(|e| Error::Parse(format!("{}: {e}", path.display())))?;
                self.merge_file(file)?;
            }
        }
        Ok(())
    }

    fn merge_file(&mut self, file: LibraryFile) -> Result<()> {
        if let Some(r) = file.recipe {
            self.recipe = r;
        }
        for (name, e) in file.decrees {
            let d = Decree {
                name: name.clone(),
                question: e.question,
                options: e.options,
                kind: e.kind,
                abstain: e.abstain,
                on_abstain: e.on_abstain,
                default_id: e.default_id,
                costs: e.costs,
                tests: e.tests,
                variants: e.variants,
                backend: e.backend,
                debias: e.debias,
                tree: e.tree,
                recipe: Some(self.recipe.clone()),
            };
            if self.decrees.insert(name.clone(), d).is_some() {
                return Err(Error::Validation(format!("duplicate decree {name}")));
            }
        }
        for (name, mut p) in file.programs {
            p.name = name.clone();
            if self.programs.insert(name.clone(), p).is_some() {
                return Err(Error::Validation(format!("duplicate program {name}")));
            }
        }
        Ok(())
    }

    fn merge(&mut self, other: Library) -> Result<()> {
        if other.recipe != crate::types::PROMPT_VERSION || self.recipe == crate::types::PROMPT_VERSION
        {
            if !other.recipe.is_empty() {
                self.recipe = other.recipe;
            }
        }
        for (k, v) in other.decrees {
            if self.decrees.insert(k.clone(), v).is_some() {
                return Err(Error::Validation(format!("duplicate decree {k}")));
            }
        }
        for (k, v) in other.programs {
            if self.programs.insert(k.clone(), v).is_some() {
                return Err(Error::Validation(format!("duplicate program {k}")));
            }
        }
        Ok(())
    }
}

pub fn bind_state(input: &Value, facts: &BTreeMap<String, String>) -> Value {
    let mut map = serde_json::Map::new();
    map.insert("input".into(), input.clone());
    let mut f = serde_json::Map::new();
    for (k, v) in facts {
        f.insert(k.clone(), Value::String(v.clone()));
    }
    map.insert("facts".into(), Value::Object(f));
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn toml_round_minimal() {
        let src = r#"
schema = "ereshkigal.decree/v1"
recipe = "direct-options-v1"

[decrees.deploy_ok]
question = "Did deploy succeed?"
options = [
  { id = "yes", description = "yes" },
  { id = "no", description = "no" },
]
"#;
        let file: LibraryFile = toml::from_str(src).unwrap();
        let mut lib = Library::default();
        lib.merge_file(file).unwrap();
        assert_eq!(lib.decree("deploy_ok").unwrap().options.len(), 2);
        let row = lib
            .decree("deploy_ok")
            .unwrap()
            .to_row("t", json!("ok"))
            .unwrap();
        assert_eq!(row.question.contains("deploy"), true);
    }

    #[test]
    fn load_from_toml_str_parses() {
        let src = r#"
recipe = "direct-options-v1"
[decrees.x]
question = "Q?"
options = [
  { id = "a", description = "A" },
  { id = "b", description = "B" },
]
"#;
        let lib = Library::load_from_toml_str(src).unwrap();
        assert!(lib.decree("x").is_ok());
    }
}
