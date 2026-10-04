(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ereshkigalPrompt = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const LETTERS = "ABCDEFGHIJKLMNOP";
  const SYSTEM = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
  function dumps(v) {
    if (v === null) return "null";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v === "number") return String(v);
    if (typeof v === "string") return JSON.stringify(v);
    if (Array.isArray(v)) return "[" + v.map(dumps).join(", ") + "]";
    const keys = Object.keys(v);
    return "{" + keys.map((k) => JSON.stringify(k) + ": " + dumps(v[k])).join(", ") + "}";
  }
  function renderDirect(state, question, options) {
    const opts = options.map((description, i) => ({
      letter: LETTERS[i],
      description,
    }));
    const payload = dumps({ evidence: state, criterion: question, options: opts });
    return `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n${payload}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
  }
  async function sha256Hex(text) {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  return { renderDirect, sha256Hex, LETTERS };
});
