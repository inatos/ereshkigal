use ereshkigal_core::{render_prompt, DecisionRow, OptionSpec};
use serde_json::json;
fn main() {
    let row = DecisionRow {
        id: "support-1".into(),
        state: json!("The deployment completed at 14:02 UTC. Health checks passed in all three zones. No rollback was initiated."),
        question: "Is there evidence that the deployment succeeded?".into(),
        options: vec![
            OptionSpec { id: "yes".into(), description: "The deployment succeeded.".into() },
            OptionSpec { id: "no".into(), description: "The deployment did not succeed.".into() },
            OptionSpec { id: "insufficient".into(), description: "The evidence is insufficient to decide.".into() },
        ],
    };
    let p = render_prompt(&row).unwrap();
    std::fs::write("/tmp/rust_prompt.txt", &p).unwrap();
    println!("{}", p);
}
