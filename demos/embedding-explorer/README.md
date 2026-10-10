# Embedding explorer

Word embeddings made from counts, in X_eTaL. 4,000 short sentences
over 70 words; each word's embedding is 64 numbers made from which
words occur near it, by the [Embed](../../libs/Embed/docs/README.md)
library. Words used alike get embeddings that point alike: the
animals gather, and the foods, the colors, the numbers, the places.
[Learn](../../libs/Learn/docs/README.md)'s PCA turns the 64 dimensions
into the 3 that spread the words most, for a cloud to turn.

Live: [Embedding explorer](https://softwarewrighter.github.io/X_eTaL-ML/embedding-explorer/)
(drag the cloud to turn it, click a word for its five nearest; the
colors are the clusters k-means finds in the 64 numbers, not told the
kinds).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                          # fetch and build the pinned X_eTaL (once)
just run embedding-explorer         # five words' nearest, and how much of the spread 3-D keeps
just serve embedding-explorer       # the web app at http://127.0.0.1:8435/
just test-demo embedding-explorer   # its CLI and browser baselines and the web app's tests
```

## The program

`embedding-explorer.xtl`, the core:

```
C := (v, 2) em:c_ooccur ids          # how often each word occurs within 2 of each other
P := em:p_pmi C                      # weighted: how much more often than chance
E := em:u_nit 64 em:p_roject P       # 64 numbers a word, each row length 1
(mu, V, ev) := 3 ml:p_ca E           # the 3 directions of most spread
xyz := (mu, V, ev) ml:p_roject E
```

Counting is by sorting: every pair of word numbers within 2 words in
a sentence becomes one key, the keys are put in order (`g_rade`), and
each run of equal keys is a count. PPMI is the log of how much more
often a pair occurs than if the words were independent, 0 where less.
The projection to 64 numbers is a fixed matrix of +1 and -1 (random
indexing, its signs from a hash of each place, so the same every
run): it roughly keeps the angles between the rows.

## What it prints

Five words' five nearest, by the cosine of their 64-number rows:

```
cat: lion dog bird fox sheep
apple: cheese rice soup egg milk
red: brown yellow black purple gray
three: two one six count seven
park: home garden city beach river
```

then the share of the spread the 3 directions keep (0.506). Words of
a kind are each other's nearest; within a kind the words' habits
order them (three is nearest two, the number it is counted with).

## How many sentences are enough

Scored by whether a content word's nearest word is of its own kind,
and how many of its 5 nearest are, on corpora from the same generator
(`train/`, which takes a sentence count and a directory):

| varied sentences | nearest of its kind | top 5 of its kind |
| ---------------: | ------------------: | ----------------: |
| 12 | 0.33 | 0.23 |
| 50 | 0.54 | 0.44 |
| 100 | 0.67 | 0.60 |
| 200 | 0.91 | 0.77 |
| 400 | 0.97 | 0.83 |
| 1,600 | 0.90 | 0.89 |
| 4,000 (this demo) | 0.90 | 0.88 |
| 16,000 | 0.90 | 0.87 |

About 1,600 is enough for 70 words: each content word has then been
seen about a hundred times in varied company. More adds nothing;
repeating sentences adds nothing at all (PPMI is a ratio). What is
left comes from the corpus's own make-up (actions share sentences
with animals and places; each word's habits mix kinds) and a little
from the projection to 64 numbers (on the PPMI rows themselves the
top-5 share at 4,000 sentences is 0.92).

## The data

`data/words.txt` (the 70 words, a line each) and `data/ids.txt` (the
sentences as word numbers, each ended by 0), written by `train/` (`just
embedding-data`): 4,000 sentences from six templates over six kinds of
words, with a fixed seed; within a kind each word has habits (an
animal's favorite food, place and action, a food's usual color,
counting mostly in order), six times in ten; `data/sample.txt` shows the first 20 as
text. The embeddings are made in X_eTaL from these, nothing trained
offline.

## Background

Count-based embeddings: the distributional hypothesis (Harris 1954,
Firth 1957: a word is known by the company it keeps), PPMI weighting
(Church and Hanks 1990; Bullinaria and Levy 2007), and random
indexing (Kanerva, Kristofersson and Holst 2000; Sahlgren 2005) to
cut the dimensions; then principal components to look at them.
