# k-means

Clustering without labels, step by step. 300 points in five blobs and
five centers: each step gives every point its nearest center, then
moves each center to the mean of its points. The step is the
[Learn](../../libs/Learn/docs/README.md) library's `ml:k_step`, a few
array expressions over all the points at once.

Live: [k-means](https://softwarewrighter.github.io/X_eTaL-ML/k-means/)
(step or run; the map shows each place's nearest center, the points in
their center's color, the centers as crosses with their paths; pick
the start and k from 2 to 8).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal               # fetch and build the pinned X_eTaL (once)
just run k-means         # the inertia step by step from two starts, the centers, the map
just show k-means        # as a notebook: each statement, then its output
just serve k-means       # the web app at http://127.0.0.1:8435/
just test-demo k-means   # its CLI and browser baselines and the web app's tests
```

## The program

`k-means.xtl` makes five blobs of 60 points in X_eTaL (each its own
center and spread, the noise from `r_oll!`, seeded), then runs
k-means from two starts:

```
C0 := k t_ake X                 # a poor start: five points of the first blob
F0 := k ml:f_arthest X          # Learn's: the first point, then each time the farthest
C10 := 10 '{ C -> X ml:k_step C } p_ower C0
```

A step, in Learn:

```
l:k_step := { X C ->
  O := (t_ally C) nn:o_neHot C l:a_ssign X
  n := ('+ r_/_1 O) 'l_eft t_able o_ffsets 2 s_elect s_hape X
  (((o_\ O) '+ '* i_nner X) + C * f_loat n = 0.0) / 1.0 m_ax n
}
```

`O` is each point's nearest center as one-hot rows; its columns times
the points sum each cluster's points, and dividing by each cluster's
count gives the means. A center with no points keeps its place.

## What it prints

1. The inertia (the sum of squared distances to the nearest center)
   after 0 to 10 steps from the poor start: 299.8 down to 50.17, where
   it stays. Three centers crowd the first blob: k-means only ever
   moves downhill, and this is a local minimum.
2. The same from farthest-first: 7.5 to 1.897, one center a blob.
3. The poor start's centers after 10 steps, and the map of which
   center each place is nearest, as letters A to E.

## Background

k-means is Lloyd's algorithm (1957); the farthest-first start is
Gonzalez's (1985), a deterministic cousin of k-means++. The demo is
the k-means of APLearn's list on the APL Wiki's machine-learning
page, reimplemented for X_eTaL in the Learn library.
