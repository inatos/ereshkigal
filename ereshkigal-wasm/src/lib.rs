use ereshkigal_lang::{digest, render_prompt, DecisionRow, Library};
use serde_json::Value;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(js_name = renderPrompt)]
pub fn render_prompt_js(row_json: &str) -> Result<String, JsValue> {
    let row: DecisionRow = serde_json::from_str(row_json).map_err(js)?;
    render_prompt(&row).map_err(js)
}

#[wasm_bindgen(js_name = promptSha256)]
pub fn prompt_sha256_js(row_json: &str) -> Result<String, JsValue> {
    let row: DecisionRow = serde_json::from_str(row_json).map_err(js)?;
    let p = render_prompt(&row).map_err(js)?;
    Ok(digest(&p))
}

#[wasm_bindgen(js_name = parseEsk)]
pub fn parse_esk(src: &str) -> Result<JsValue, JsValue> {
    let lib = ereshkigal_lang::syntax::parse_library(src).map_err(js)?;
    let names: Vec<String> = lib.decrees.keys().cloned().collect();
    serde_wasm_bindgen::to_value(&names).map_err(js)
}

#[wasm_bindgen(js_name = lowerDecree)]
pub fn lower_decree(src: &str, name: &str, state_json: &str) -> Result<String, JsValue> {
    let lib = Library::load_str_for_wasm(src).or_else(|_| {
        ereshkigal_lang::syntax::parse_library(src).map_err(js)
    })?;
    let state: Value = serde_json::from_str(state_json).map_err(js)?;
    let d = lib.decree(name).map_err(js)?;
    let row = d.to_row(name, state).map_err(js)?;
    serde_json::to_string(&row).map_err(js)
}

#[wasm_bindgen(js_name = lint)]
pub fn lint_js(src: &str) -> Result<String, JsValue> {
    match ereshkigal_lang::syntax::parse_library(src) {
        Ok(lib) => match ereshkigal_lang::lint::lint_library(&lib) {
            Ok(()) => Ok("ok".into()),
            Err(e) => Ok(e.to_string()),
        },
        Err(e) => Ok(e.to_string()),
    }
}

fn js<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// helper implemented on Library in lang — stub local
trait LoadStr {
    fn load_str_for_wasm(s: &str) -> Result<Library, JsValue>;
}
impl LoadStr for Library {
    fn load_str_for_wasm(_s: &str) -> Result<Library, JsValue> {
        Err(JsValue::from_str("use parseEsk"))
    }
}
