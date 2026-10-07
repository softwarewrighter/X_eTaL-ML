# training

Saga 4 of X_eTaL-ML (docs/plan.md): training in X_eTaL. After the APL
Wiki's neural-network resources, every one of which trains its
network in APL, while every demo here infers with weights trained
offline. Backpropagation is where array notation pays off: each
layer's gradient is one expression.

Rules as before (CLAUDE.md): pinned X_eTaL; asks filed; American
spellings; ASCII docs; programs short, data in files; concise
X_eTaL; a page shows all the code it runs. Every step: tests and
`just gate`, docs, a detailed commit, `agentrail complete`, push,
`just publish` and `just check-live` when a page changed, a report.

## Steps

1. backprop-microscope -- one training step, every array visible,
   gradients checked against finite differences; CLI and page.
2. train-live -- the spiral classifier trained in the browser.
3. net-train-macro -- net:t_rain< writes forward, gradients, update.
4. cnn-backprop -- the Tiny CNN's backward pass (after Shinkarov).
5. classic-ml -- a fit/pred library (k-means, kNN, PCA, logistic
   regression) and a k-means demo.
