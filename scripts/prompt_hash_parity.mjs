#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { pathToFileURL } = require("url");

async function main() {
  const mod = await import(pathToFileURL(path.join(__dirname, "../webgpu-demo/prompt.mjs")).href);
  const expectedPath = path.join(__dirname, "../fixtures/expected_direct.jsonl");
  const examples = path.join(__dirname, "../examples/decisions.jsonl");
  const rows = fs.readFileSync(examples, "utf8").trim().split("\n").map(JSON.parse);
  const expected = fs.existsSync(expectedPath)
    ? fs.readFileSync(expectedPath, "utf8").trim().split("\n").map(JSON.parse)
    : [];
  for (const row of rows) {
    const prompt = mod.renderDirect(
      row.state,
      row.question,
      row.options.map((o) => o.description),
    );
    const sha = crypto.createHash("sha256").update(prompt).digest("hex");
    const exp = expected.find((e) => e.id === row.id);
    if (exp && exp.prompt_sha256 !== sha) {
      console.error("hash mismatch", row.id, sha, exp.prompt_sha256);
      process.exit(1);
    }
    console.log(row.id, sha);
  }
}

main();
