# CNN training

X_eTaL trains a tiny convolutional network, from random weights, on
600 handwritten digits: 8 filters of 3 x 3, ReLU, 2 x 2 max-pooling, a
dense layer and softmax, the network the [Tiny CNN](../cnn-digits/README.md)
demo reads digits with (there trained offline). Here the backward pass
is X_eTaL too: it undoes each stage in turn, a few array expressions
each over the whole batch, and is checked against finite
differences. Adam takes 20 digits a step; the 200 test digits, never
trained on, go from 12% read right to about 90% in 40 steps.

Live: [CNN training](https://softwarewrighter.github.io/X_eTaL-ML/cnn-backprop/)
(press Train: two steps a run, the filters, the first 20 test digits'
readings and the test loss and accuracy redrawn after each, the chart
by the [Plot](https://github.com/softwarewrighter/X_eTaL-libraries)
library in the page's program).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                     # fetch and build the pinned X_eTaL (once)
just run cnn-backprop          # the gradient check, the test loss and accuracy as it trains, 20 readings
just show cnn-backprop         # as a notebook: each statement, then its output
just serve cnn-backprop        # the web app at http://127.0.0.1:8435/
just test-demo cnn-backprop    # its CLI and browser baselines and the web app's tests
```

## The program

`cnn-backprop.xtl`. Forward, for a batch x of n digits: every 3 x 3
window of every digit as 9 rows (`u:w_indows`, the picture's nine
shifted copies), the filters over all of them as one matrix product
(`u:c_onv`), ReLU, the largest of each 2 x 2 block (`u:p_ool`), then
the dense layer and softmax. Backward, each stage undone in reverse
(`u:g_rad`):

```
D3 := (P - y) / f_loat n                                    # softmax with cross-entropy
GW := (o_\ F c_at_2 ((n c_at 1) r_eshape 1.0)) '+ '* i_nner D3   # the dense layer's gradient
DQ := (n c_at 8 13 13) r_eshape D3 '+ '* i_nner o_\ -1 d_rop W  # back through it
R := u:b_locks A                                            # each 2 x 2 block as a last axis of 4
hit := f_loat R = (s_hape R) r_eshape (r_avel Q) 'l_eft t_able 1 2 3 4
first := hit * f_loat 1.0 = '+ s_\_5 hit                    # the block's first largest item
DC := (u:m_aps first * ...) * f_loat C > 0.0                # unpooled, back through ReLU
DM := (8 c_at n * 676) r_eshape 2 1 3 t_ranspose (n c_at 8 c_at 676) r_eshape DC
(DM '+ '* i_nner o_\ V, '+ r_/_2 DM, GW)                    # the filters', the biases', the dense layer's
```

The pooling sends each block's gradient to its largest item, and to
the first one only when several tie: MNIST's saturated strokes make
equal maxima common, and the loss moves with one of them, not all.
The filters' gradient uses the same windows V as the forward pass,
transposed: what came back to each position, times the window that
produced it, summed over every position of every digit in one matrix
product.

The training state is a tuple, `(K, B, W, MK, MB, MW, VK, VB, VW, t)`
(the filters, their biases, the dense weights, Adam's two averages of
each, the step count), which `u:s_tep` takes apart by name and
`p_ower` repeats: `s1 := 10 'u:s_tep p_ower s0`.

## What it prints

1. The gradient check: the backward pass's gradients for the first
   filter's 9 weights, the 8 biases and 6 dense weights against
   central finite differences on 4 digits, all within 1e-8: `1`.
2. The test loss and share right at the start, after 10 steps and
   after 40: from 2.625 and 12.5% to 1.129 and 68%, then 0.322 and 91%.
3. What the trained network reads in the first 20 test digits, then
   their labels (it misses a 5, a 5 and a 3 here).

The filters' biases start at 0.01, not 0: a blank window's output is
then just above ReLU's kink, where a finite difference measures the
slope the backward pass uses.

## The data

`data/train.txt` and `data/test.txt`: the first 600 digits of MNIST's
training set and the first 200 of its test set, one per line, the
label then 784 pixels (0 to 255). `train/` writes them from MNIST
(LeCun, Cortes and Burges), downloaded into the gitignored `work/` by
`just cnn-backprop-data`; the network itself is trained here, in
X_eTaL, from weights `r_oll!` draws (seeded: the same run each time).

## Speed

A step on 20 digits takes about 0.4 s at the command line and about
1.6 s in the browser, where each run also tests on the 200 test
digits. Most of it is two matrix products of about a million
multiply-adds each (the convolution and the filters' gradient), at
about 85 ns a multiply-add (ask M16 in
[`docs/xetal-asks.md`](../../docs/xetal-asks.md)); so the demo uses 600
digits and 20 a step, not MNIST's 60,000.

## Background

After Artem Shinkarov's "Convolutional neural networks in APL" (2019,
on the APL Wiki), which trains a CNN on MNIST in APL with each layer's
backward pass a short expression; this one is reimplemented for
X_eTaL's own operations (the convolution as shifted copies times the
filters, the pooling's blocks reshaped to a last axis), not ported.
