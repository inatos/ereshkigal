module.exports = grammar({
  name: "ereshkigal",
  extras: $ => [/\s/, $.comment],
  rules: {
    source_file: $ => repeat(choice($.recipe, $.decree, $.program)),
    comment: _ => token(seq("//", /.*/)),
    recipe: $ => seq("recipe", $.ident),
    decree: $ => seq("decree", $.ident, $.string, "{", repeat($.decree_item), "}"),
    decree_item: $ => choice(
      seq($.ident, $.string),
      seq("abstain", optional(seq("coverage", $.number)), optional(seq("=>", $.ident))),
      seq("test", optional($.ident), optional(seq("group", $.string)), $.string, "=>", $.ident)
    ),
    program: $ => seq("program", $.ident, optional(seq("(", $.ident, ")")), "{", repeat($.stmt), "}"),
    stmt: $ => choice(
      seq("let", $.ident, "=", $.ident, optional($.foreach)),
      seq("match", $.ident, "{", repeat($.ident), "}")
    ),
    foreach: $ => choice(
      seq("filter", $.ident),
      seq("group", $.ident),
      seq("top", $.number, "of", $.ident),
      seq("sort", $.ident, optional("pairwise"))
    ),
    ident: _ => /[A-Za-z_][A-Za-z0-9_-]*/,
    string: _ => /"([^"\\]|\\.)*"/,
    number: _ => /[0-9]+(\.[0-9]+)?/,
  },
});
