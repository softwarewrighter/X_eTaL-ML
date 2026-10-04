# Tiny CNN

A tiny convolutional network reads a handwritten digit: 28 x 28 ->
8 filters of 3 x 3 (8 x 26 x 26) -> ReLU -> 2 x 2 max-pooling
(8 x 13 x 13) -> dense (10) -> softmax. Every stage is an array
program over the whole picture; the convolution is the picture's nine
shifted copies times the filters, one matrix product.

The web app (draw a digit, see every stage) is the next step; the
command-line program classifies ten MNIST test digits.

## Recorded at the command line

![cnn-digits at the command line: just tour cnn-digits](recording.webp)

`just tour cnn-digits`: each statement of the program as X_eTaL draws it,
then its result (long lines clipped, runs of weight literals
collapsed), recorded with VHS (`cnn-digits.tape`, `just record cnn-digits`).
Demos that run only at the command line are shown recorded (the
interactive ones are live), with how to run them yourself:
[Recorded CLI demos](https://softwarewrighter.github.io/X_eTaL-ML/recorded/).

## The program

From `cnn-digits.xtl`:

```
u:w_indows := { x -> -1 0 1 o_-_2 -1 0 1 o_-_2 x }
k9 := (8 c_at 9) r_eshape k
u:c_onv := { x ->
  v := 1 d_rop_4 -1 d_rop_4 1 d_rop_3 -1 d_rop_3 u:w_indows x
  y := k9 '+ '* i_nner (9 c_at 676) r_eshape v
  (8 c_at 26 c_at 26) r_eshape y + bc 'l_eft t_able o_ffsets 676
}
u:r_elu := { x -> 0.0 m_ax x }
u:p_ool := { x -> 'm_ax r_/_3 'm_ax r_/_5 (8 c_at 13 c_at 2 c_at 13 c_at 2) r_eshape x }
u:d_ense := { h -> bd + r_avel ((1 c_at 1352) r_eshape r_avel h) '+ '* i_nner w }
u:s_oftmax := { z ->
  e := e_xp z - 'm_ax r_/ z
  e / '+ r_/ e
}
u:c_lassify := { x -> u:s_oftmax u:d_ense u:p_ool u:r_elu u:c_onv x }
```

## The weights

`train/` (a small Rust program with no dependencies) trains the
network on the 60,000 MNIST training digits (SGD with momentum,
cross-entropy, 3 passes, a fixed shuffle: the same weights every run,
in about 12 seconds) and writes into `cnn-digits.xtl`, as literals:
the 8 filters and their biases, the 1352 x 10 dense weights and
biases, and ten test digits, one of each class. The written weights
(rounded to 5 decimals) classify 9782 of the 10,000 MNIST test digits
(97.82%) right; the X_eTaL program reads all ten sample digits right.

To retrain:

```bash
just cnn-train          # fetch MNIST (scripts/mnist.sh), train, write the weights
just bless cnn-digits   # accept the program's new output (review the diff)
```

`scripts/mnist.sh` downloads the four MNIST files (LeCun, Cortes and
Burges) from the PyTorch project's public mirror into `work/mnist/`
(not committed), checks their MD5 sums and unpacks them.

## Run it

```bash
just run cnn-digits          # a test digit, the stages' shapes, its probabilities, the ten digits read
just tour cnn-digits         # each statement, then its result, paced
just show cnn-digits         # the same as a notebook
just test-demo cnn-digits    # its CLI baseline (reg-rs)
```

## Workarounds

The weights are literals in the program (about 13,600 numbers). At
the command line they could be a data file read with
`n_umbers []N_GET "..."`; literals keep one program that runs
unchanged in the browser, where the page's X_eTaL has no files. The dense layer is a
matrix product of a 1 x 1352 row, slow per multiply-add (an ask in
[`docs/xetal-asks.md`](../../docs/xetal-asks.md)), though fast enough
here.
