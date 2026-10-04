const { LanguageClient, TransportKind } = require("vscode-languageclient/node");
const vscode = require("vscode");

let client;
function activate(context) {
  const server = { command: "ereshkigal", args: ["lsp"], transport: TransportKind.stdio };
  client = new LanguageClient("ereshkigal", "Ereshkigal LSP", { run: server, debug: server }, {
    documentSelector: [{ language: "ereshkigal" }],
  });
  context.subscriptions.push(client.start());
}
function deactivate() {
  return client ? client.stop() : undefined;
}
module.exports = { activate, deactivate };
