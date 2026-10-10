# Embed

Word embeddings from counts: which words occur near which, weighted
by how much more often than chance (PPMI), projected to fewer
numbers, compared by cosine. Every step is whole-array arithmetic;
the counting is a sort.

```
"em:" u_se< "Embed"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `em:` is the
recommended alias. Embed imports Lists from X_eTaL-libraries (for
`r_aze`). The demos and live pages of this repository find the
libraries without setup.

## Conventions

- A corpus is a vector of word numbers from 1, each sentence ended by
  a 0; words are counted together only within a sentence.
- A word's embedding is a row; a table of embeddings is a matrix, a
  row per word.
- Nothing draws on chance: the projection's signs come from a hash of
  each place.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `n em:c_ounts k` | `Int -> Int -> Int` | how many times each of 1 .. n occurs in k, by sorting (the keys in order, then each run's length) |
| `(v, w) em:c_ooccur ids` | `(Int, Int) -> Int -> Int` | a v x v table of how often word i occurs within w words of word j in a sentence, both ways |
| `em:p_pmi C` | `Num a => a -> Float` | positive pointwise mutual information of a count table: log of observed over expected, 0 where less or never |
| `d em:p_roject M` | `Int -> Float -> Float` | each row to d numbers by a fixed matrix of +1 and -1 over the square root of d (random indexing) |
| `em:u_nit E` | `Float -> Float` | each row scaled to length 1 |
| `E em:c_osines i` | `Float -> Int -> Float` | the cosine of row i with every row |
| `(E, k) em:n_earest i` | `(Float, Int) -> Int -> Int` | the k rows most like row i, nearest first, row i left out |

## Examples

From `../tests/basics.xtl`: two sentences, "1 2 3" and "1 2", and who
occurs within 1 word of whom:

```
      5 em:c_ounts 3 1 3 3 5
1 0 3 0 1
      C := (3, 1) em:c_ooccur 1 2 3 0 1 2 0
      C
0 2 0
2 0 1
0 1 0
```

`../tests/checks.xtl` checks the properties on a corpus of two kinds
of word: the counts are symmetric, `c_ounts` counts every key once,
PPMI is never below 0 and 0 where the count is, the projection is the
same every run, rows come out length 1, and each word's nearest are
of its own kind. The `## >>` examples are run by `just doc-test`.

## Demos

- [`demos/pets.xtl`](../demos/pets.xtl): twelve sentences about pets
  and food, small enough to read the counts and PPMI.
- [`../../../demos/embedding-explorer`](../../../demos/embedding-explorer/README.md):
  70 words, 4,000 sentences, a 3-D cloud to turn.

## Limits

- The projection keeps angles only roughly (more numbers, less
  error); it is chosen because it is one matrix product, where a
  64-direction PCA of the PPMI table would take power iteration 64
  times.
- Counting holds every pair in one sorted vector: a corpus of a few
  hundred thousand words is fine; much more wants counting by parts.

## Provenance

Written for X_eTaL-ML. The methods are the textbook ones: the
distributional hypothesis (Harris 1954; Firth 1957), PPMI (Church and
Hanks 1990; Bullinaria and Levy 2007), random indexing (Kanerva,
Kristofersson and Holst 2000; Sahlgren 2005); counting by sorting
then run lengths is APL's own idiom (as Lists' run lengths).
