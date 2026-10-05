# `.esk` grammar (EBNF)

```
library     = { recipe | decree | program } ;
recipe      = "recipe" ident ;
decree      = "decree" ident string "{" { option | abstain | cost | test | kind | backend } "}" ;
option      = ident string ;
abstain     = "abstain" [ "coverage" number ] [ "=>" ( "return" | "escalate" | "fail" | ident ) ] ;
cost        = "cost" ident "->" ident "=" number ;
test        = "test" [ "dev" | "test" ] [ "group" string ] string "=>" ident ;
pass     = { filter | topk | group | sort } ;
filter    = "filter" ident ;
group     = "group" ident ;
topk      = "top" number "of" ident ;
sort      = "sort" ident [ "pairwise" ] ;
```

Parser: logos + recursive descent (`ereshkigal-lang::syntax`). Formatter: `ereshkigal fmt`.
