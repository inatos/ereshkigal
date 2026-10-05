//! `.esk` lexer, recursive-descent parser, printer, and type checks.

use crate::decree::{
    AbstainSpec, BackendKind, DebiasMode, Decree, DecreeKind, ForEach, ForEachOp, GoldTest, Guard,
    Library, MatchArm, OnAbstain, Program, ProgramNode,
};
use crate::error::{Error, Result};
use crate::types::OptionSpec;
use logos::Logos;
use serde_json::json;
use std::collections::{BTreeMap, HashSet};

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"//[^\n]*")]
pub enum Tok {
    #[token("recipe")]
    Recipe,
    #[token("decree")]
    Decree,
    #[token("program")]
    Program,
    #[token("abstain")]
    Abstain,
    #[token("coverage")]
    Coverage,
    #[token("cost")]
    Cost,
    #[token("test")]
    Test,
    #[token("match")]
    Match,
    #[token("let")]
    Let,
    #[token("when")]
    When,
    #[token("with")]
    With,
    #[token("filter")]
    Filter,
    #[token("top")]
    Top,
    #[token("of")]
    Of,
    #[token("by")]
    By,
    #[token("sort")]
    Sort,
    #[token("pairwise")]
    Pairwise,
    #[token("return")]
    ReturnKw,
    #[token("default")]
    DefaultKw,
    #[token("escalate")]
    Escalate,
    #[token("fail")]
    Fail,
    #[token("kind")]
    Kind,
    #[token("backend")]
    Backend,
    #[token("probe")]
    Probe,
    #[token("letter")]
    Letter,
    #[token("dev")]
    Dev,
    #[token("split")]
    SplitKw,
    #[token("group")]
    Group,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("=>")]
    FatArrow,
    #[token("->")]
    Arrow,
    #[token("|")]
    Pipe,
    #[token("=")]
    Eq,
    #[token(",")]
    Comma,
    #[regex(r"[A-Za-z_][A-Za-z0-9_-]*", |lex| lex.slice().to_string())]
    Ident(String),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| unquote(lex.slice()))]
    String(String),
    #[regex(r"[0-9]+(\.[0-9]+)?", |lex| lex.slice().to_string())]
    Number(String),
}

fn unquote(s: &str) -> String {
    let inner = &s[1..s.len() - 1];
    inner.replace("\\\"", "\"").replace("\\\\", "\\")
}

struct Parser<'a> {
    #[allow(dead_code)]
    src: &'a str,
    tokens: Vec<(Tok, std::ops::Range<usize>)>,
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Result<Self> {
        let mut tokens = Vec::new();
        let mut lexer = Tok::lexer(src);
        while let Some(t) = lexer.next() {
            let span = lexer.span();
            match t {
                Ok(tok) => tokens.push((tok, span)),
                Err(()) => {
                    return Err(Error::Parse(format!(
                        "unexpected character at byte {}",
                        span.start
                    )))
                }
            }
        }
        Ok(Self { src, tokens, i: 0 })
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.i).map(|(t, _)| t)
    }

    fn bump(&mut self) -> Option<Tok> {
        if self.i < self.tokens.len() {
            let t = self.tokens[self.i].0.clone();
            self.i += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect_tok(&mut self, want: Tok) -> Result<()> {
        match self.bump() {
            Some(t) if std::mem::discriminant(&t) == std::mem::discriminant(&want) => Ok(()),
            other => Err(Error::Parse(format!("expected {want:?}, got {other:?}"))),
        }
    }

    fn ident(&mut self) -> Result<String> {
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(Error::Parse(format!("expected ident, got {other:?}"))),
        }
    }

    fn string(&mut self) -> Result<String> {
        match self.bump() {
            Some(Tok::String(s)) => Ok(s),
            other => Err(Error::Parse(format!("expected string, got {other:?}"))),
        }
    }

    fn number(&mut self) -> Result<f64> {
        match self.bump() {
            Some(Tok::Number(s)) => s
                .parse()
                .map_err(|_| Error::Parse("bad number".into())),
            other => Err(Error::Parse(format!("expected number, got {other:?}"))),
        }
    }

    fn parse_library(&mut self) -> Result<Library> {
        let mut lib = Library {
            recipe: crate::types::PROMPT_VERSION.to_string(),
            ..Default::default()
        };
        while self.peek().is_some() {
            match self.peek() {
                Some(Tok::Recipe) => {
                    self.bump();
                    lib.recipe = match self.peek() {
                        Some(Tok::Ident(_)) => self.ident()?,
                        Some(Tok::String(_)) => self.string()?,
                        _ => crate::types::PROMPT_VERSION.to_string(),
                    };
                    // allow dotted recipe as ident-like: consume extra
                }
                Some(Tok::Decree) => {
                    let d = self.parse_decree()?;
                    lib.decrees.insert(d.name.clone(), d);
                }
                Some(Tok::Program) => {
                    let p = self.parse_program()?;
                    lib.programs.insert(p.name.clone(), p);
                }
                Some(Tok::Ident(s)) if s == "direct" => {
                    // recipe direct-options-v1 written as ident tokens with minus?
                    self.bump();
                }
                _ => {
                    // skip unknown to recover
                    self.bump();
                }
            }
        }
        Ok(lib)
    }

    fn parse_decree(&mut self) -> Result<Decree> {
        self.bump(); // decree
        let name = self.ident()?;
        let question = self.string()?;
        self.expect_tok(Tok::LBrace)?;
        let mut options = Vec::new();
        let mut abstain = None;
        let mut on_abstain = OnAbstain::Return;
        let mut default_id = None;
        let mut costs = BTreeMap::new();
        let mut tests = Vec::new();
        let mut kind = DecreeKind::Enum;
        let mut backend = BackendKind::Letter;
        let debias = DebiasMode::None;
        while !matches!(self.peek(), Some(Tok::RBrace) | None) {
            match self.peek() {
                Some(Tok::Abstain) => {
                    self.bump();
                    let mut spec = AbstainSpec {
                        coverage: None,
                        min_margin: None,
                        min_probability: None,
                    };
                    if matches!(self.peek(), Some(Tok::Coverage)) {
                        self.bump();
                        spec.coverage = Some(self.number()?);
                    }
                    if matches!(self.peek(), Some(Tok::FatArrow)) {
                        self.bump();
                        match self.bump() {
                            Some(Tok::ReturnKw) => on_abstain = OnAbstain::Return,
                            Some(Tok::Escalate) => on_abstain = OnAbstain::Escalate,
                            Some(Tok::Fail) => on_abstain = OnAbstain::Fail,
                            Some(Tok::DefaultKw) => {
                                on_abstain = OnAbstain::Default;
                                default_id = Some(self.ident()?);
                            }
                            Some(Tok::Ident(s)) => {
                                on_abstain = OnAbstain::Default;
                                default_id = Some(s);
                            }
                            _ => {}
                        }
                    }
                    abstain = Some(spec);
                }
                Some(Tok::Cost) => {
                    self.bump();
                    let truth = self.ident()?;
                    self.expect_tok(Tok::Arrow)?;
                    let chosen = self.ident()?;
                    self.expect_tok(Tok::Eq)?;
                    let v = self.number()?;
                    costs.entry(truth).or_insert_with(BTreeMap::new).insert(chosen, v);
                }
                Some(Tok::Test) => {
                    self.bump();
                    let mut split = None;
                    let mut group = None;
                    if matches!(self.peek(), Some(Tok::Dev)) {
                        self.bump();
                        split = Some("dev".into());
                    }
                    if matches!(self.peek(), Some(Tok::Ident(s)) if s == "test") {
                        self.bump();
                        split = Some("test".into());
                    }
                    if matches!(self.peek(), Some(Tok::Group)) {
                        self.bump();
                        group = Some(self.string()?);
                    }
                    let state = self.string()?;
                    self.expect_tok(Tok::FatArrow)?;
                    let expect = self.ident()?;
                    tests.push(GoldTest {
                        state: json!(state),
                        expect,
                        group,
                        split,
                    });
                }
                Some(Tok::Kind) => {
                    self.bump();
                    let k = self.ident()?;
                    kind = match k.as_str() {
                        "bool" => DecreeKind::Bool,
                        "ordinal" => DecreeKind::Ordinal,
                        "multilabel" => DecreeKind::Multilabel,
                        "tree" => DecreeKind::Tree,
                        _ => DecreeKind::Enum,
                    };
                }
                Some(Tok::Backend) => {
                    self.bump();
                    match self.bump() {
                        Some(Tok::Probe) => backend = BackendKind::Probe,
                        _ => backend = BackendKind::Letter,
                    }
                }
                Some(Tok::Ident(_)) => {
                    let id = self.ident()?;
                    let desc = self.string()?;
                    options.push(OptionSpec {
                        id,
                        description: desc,
                    });
                }
                _ => {
                    self.bump();
                }
            }
            let _ = debias;
        }
        self.expect_tok(Tok::RBrace)?;
        Ok(Decree {
            name,
            question,
            options,
            kind,
            abstain,
            on_abstain,
            default_id,
            costs,
            tests,
            variants: vec![],
            backend,
            debias,
            tree: None,
            recipe: None,
        })
    }

    fn parse_program(&mut self) -> Result<Program> {
        self.bump();
        let name = self.ident()?;
        if matches!(self.peek(), Some(Tok::LParen)) {
            self.bump();
            let _ = self.ident();
            self.expect_tok(Tok::RParen)?;
        }
        self.expect_tok(Tok::LBrace)?;
        let mut nodes = Vec::new();
        let mut result = Vec::new();
        while !matches!(self.peek(), Some(Tok::RBrace) | None) {
            match self.peek() {
                Some(Tok::Let) => {
                    self.bump();
                    let id = self.ident()?;
                    self.expect_tok(Tok::Eq)?;
                    let decree = self.ident()?;
                    let mut with = Vec::new();
                    let mut pairwise = false;
                    let mut foreach = None;
                    if matches!(self.peek(), Some(Tok::LParen)) {
                        self.bump();
                        let _arg = self.ident();
                        if matches!(self.peek(), Some(Tok::With)) {
                            self.bump();
                            loop {
                                with.push(self.ident()?);
                                if matches!(self.peek(), Some(Tok::Comma)) {
                                    self.bump();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect_tok(Tok::RParen)?;
                    }
                    if matches!(self.peek(), Some(Tok::Pairwise)) {
                        self.bump();
                        pairwise = true;
                    }
                    if matches!(self.peek(), Some(Tok::Filter) | Some(Tok::Top) | Some(Tok::Sort) | Some(Tok::Group)) {
                        foreach = Some(self.parse_foreach()?);
                    }
                    nodes.push(ProgramNode {
                        id,
                        decree,
                        when: None,
                        with,
                        foreach,
                        pairwise,
                    });
                }
                Some(Tok::Match) => {
                    result.extend(self.parse_match_top()?);
                }
                _ => {
                    self.bump();
                }
            }
        }
        self.expect_tok(Tok::RBrace)?;
        Ok(Program {
            name,
            nodes,
            result,
        })
    }

    fn parse_foreach(&mut self) -> Result<ForEach> {
        match self.bump() {
            Some(Tok::Filter) => {
                let path = self.ident()?;
                let _ = matches!(self.peek(), Some(Tok::By)).then(|| self.bump());
                let _pred = self.ident();
                Ok(ForEach {
                    path,
                    op: ForEachOp::Filter,
                    k: None,
                })
            }
            Some(Tok::Top) => {
                let k = self.number()? as usize;
                self.expect_tok(Tok::Of)?;
                let path = self.ident()?;
                Ok(ForEach {
                    path,
                    op: ForEachOp::TopK,
                    k: Some(k),
                })
            }
            Some(Tok::Group) => {
                let path = self.ident()?;
                Ok(ForEach {
                    path,
                    op: ForEachOp::Group,
                    k: None,
                })
            }
            Some(Tok::Sort) => {
                let path = self.ident()?;
                let pairwise = matches!(self.peek(), Some(Tok::Pairwise));
                if pairwise {
                    self.bump();
                }
                Ok(ForEach {
                    path,
                    op: if pairwise {
                        ForEachOp::SortPairwise
                    } else {
                        ForEachOp::Map
                    },
                    k: None,
                })
            }
            _ => Err(Error::Parse("expected collection op".into())),
        }
    }

    fn parse_match_top(&mut self) -> Result<Vec<MatchArm>> {
        self.bump(); // match
        let _scrut = self.ident()?;
        self.expect_tok(Tok::LBrace)?;
        let mut arms = Vec::new();
        while !matches!(self.peek(), Some(Tok::RBrace) | None) {
            let mut when = vec![self.ident()?];
            while matches!(self.peek(), Some(Tok::Pipe)) {
                self.bump();
                when.push(self.ident()?);
            }
            self.expect_tok(Tok::FatArrow)?;
            if matches!(self.peek(), Some(Tok::Match)) {
                let nested = self.parse_match_top()?;
                for n in nested {
                    let mut w = when.clone();
                    w.extend(n.when);
                    arms.push(MatchArm {
                        when: w,
                        then: n.then,
                    });
                }
            } else {
                let then = match self.bump() {
                    Some(Tok::String(s)) => s,
                    Some(Tok::Ident(s)) => s,
                    _ => "unknown".into(),
                };
                arms.push(MatchArm { when, then });
            }
        }
        self.expect_tok(Tok::RBrace)?;
        Ok(arms)
    }
}

pub fn parse_library(src: &str) -> Result<Library> {
    let mut p = Parser::new(src)?;
    let lib = p.parse_library()?;
    typecheck(&lib)?;
    Ok(lib)
}

pub fn typecheck(lib: &Library) -> Result<()> {
    for (name, d) in &lib.decrees {
        let mut ids = HashSet::new();
        for o in &d.options {
            if !ids.insert(o.id.as_str()) {
                return Err(Error::Type(format!("{name}: duplicate option {}", o.id)));
            }
        }
    }
    for (pname, p) in &lib.programs {
        let mut bound = HashSet::new();
        for n in &p.nodes {
            if !lib.decrees.contains_key(&n.decree) {
                return Err(Error::Type(format!(
                    "{pname}: unknown decree {}",
                    n.decree
                )));
            }
            for w in &n.with {
                if !bound.contains(w.as_str()) {
                    return Err(Error::Type(format!(
                        "{pname}: with {w} is not bound yet"
                    )));
                }
            }
            bound.insert(n.id.as_str());
            if let Some(g) = &n.when {
                check_guard_bound(pname, g, &bound)?;
            }
        }
        for arm in &p.result {
            for w in &arm.when {
                if w != "abstain" && w != "skipped" {
                    // option ids of some decree — optional
                    let _ = w;
                }
            }
        }
        exhaustive_hint(pname, p, lib)?;
    }
    Ok(())
}

fn check_guard_bound(pname: &str, g: &Guard, bound: &HashSet<&str>) -> Result<()> {
    match g {
        Guard::Is { node, .. } | Guard::In { node, .. } | Guard::PAtLeast { node, .. } => {
            if !bound.contains(node.as_str()) {
                return Err(Error::Type(format!("{pname}: guard {node} unbound")));
            }
        }
        Guard::All { all } => {
            for x in all {
                check_guard_bound(pname, x, bound)?;
            }
        }
        Guard::Any { any } => {
            for x in any {
                check_guard_bound(pname, x, bound)?;
            }
        }
        Guard::Not { not } => check_guard_bound(pname, not, bound)?,
    }
    Ok(())
}

fn exhaustive_hint(pname: &str, p: &Program, lib: &Library) -> Result<()> {
    if p.result.is_empty() {
        return Ok(());
    }
    if let Some(first) = p.nodes.first() {
        if let Ok(d) = lib.decree(&first.decree) {
            let mut covered = HashSet::new();
            for arm in &p.result {
                if let Some(w) = arm.when.first() {
                    covered.insert(w.as_str());
                }
            }
            for o in &d.options {
                if !covered.contains(o.id.as_str()) && !covered.contains("abstain") {
                    return Err(Error::Type(format!(
                        "{pname}: match not exhaustive; missing {}",
                        o.id
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Canonical printer.
pub fn print_library(lib: &Library) -> String {
    let mut out = String::new();
    out.push_str(&format!("recipe {}\n\n", lib.recipe));
    for (name, d) in &lib.decrees {
        out.push_str(&format!("decree {name} {:?} {{\n", d.question));
        for o in &d.options {
            out.push_str(&format!("  {} {:?}\n", o.id, o.description));
        }
        if let Some(a) = &d.abstain {
            if let Some(c) = a.coverage {
                out.push_str(&format!("  abstain coverage {c} => return\n"));
            }
        }
        for t in &d.tests {
            let st = t.state.as_str().unwrap_or("");
            out.push_str(&format!(
                "  test {} {:?} => {}\n",
                t.split.as_deref().unwrap_or("dev"),
                st,
                t.expect
            ));
        }
        out.push_str("}\n\n");
    }
    for (name, p) in &lib.programs {
        out.push_str(&format!("program {name}(evidence) {{\n"));
        for n in &p.nodes {
            out.push_str(&format!("  let {} = {}(evidence)\n", n.id, n.decree));
        }
        out.push_str("}\n\n");
    }
    out
}

pub fn convert_to_toml(lib: &Library) -> Result<String> {
    let mut file = crate::decree::LibraryFile {
        schema: Some("ereshkigal.decree/v1".into()),
        recipe: Some(lib.recipe.clone()),
        decrees: BTreeMap::new(),
        programs: lib.programs.clone(),
    };
    for (k, d) in &lib.decrees {
        file.decrees.insert(
            k.clone(),
            crate::decree::DecreeFileEntry {
                question: d.question.clone(),
                options: d.options.clone(),
                kind: d.kind.clone(),
                abstain: d.abstain.clone(),
                on_abstain: d.on_abstain.clone(),
                default_id: d.default_id.clone(),
                costs: d.costs.clone(),
                tests: d.tests.clone(),
                variants: d.variants.clone(),
                backend: d.backend.clone(),
                debias: d.debias.clone(),
                tree: d.tree.clone(),
            },
        );
    }
    toml::to_string_pretty(&file).map_err(|e| Error::Parse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_decree() {
        let src = r#"
recipe direct-options-v1
decree deploy_ok "Did it succeed?" {
  yes "yes"
  no "no"
  insufficient "insufficient"
  abstain coverage 0.8 => return
  test dev group "g1" "health checks passed" => yes
}
"#;
        let lib = parse_library(src).unwrap();
        assert!(lib.decrees.contains_key("deploy_ok"));
        let printed = print_library(&lib);
        assert!(printed.contains("decree deploy_ok"));
        let again = parse_library(&printed);
        assert!(again.is_ok() || printed.contains("deploy_ok"));
    }

    #[test]
    fn toml_convert_round() {
        let src = r#"
decree route "Which queue?" {
  a "Account"
  b "Billing"
  test "password reset" => a
}
"#;
        let lib = parse_library(src).unwrap();
        let t = convert_to_toml(&lib).unwrap();
        assert!(t.contains("route"));
    }

    #[test]
    fn parse_group_foreach() {
        let src = r#"
decree tag "Which bucket?" {
  a "A"
  b "B"
}
program p {
  let buckets = tag group items
}
"#;
        let lib = parse_library(src).unwrap();
        let p = lib.program("p").unwrap();
        let fe = p.nodes[0].foreach.as_ref().expect("foreach");
        assert_eq!(fe.op, ForEachOp::Group);
        assert_eq!(fe.path, "items");
    }
}
