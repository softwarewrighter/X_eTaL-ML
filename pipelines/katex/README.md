# KaTeX

[KaTeX](https://katex.org) 0.18.10 (MIT, `LICENSE`), as built: `katex.min.js`,
`katex.min.css` and the WOFF2 fonts (browsers take WOFF2 first; the
stylesheet's WOFF and TTF fallbacks are not copied). A page that draws
LaTeX (the pipelines demo's equation) copies this directory with trunk
(`<link data-trunk rel="copy-dir" href="../../../shared/katex" />`) and
loads `katex/katex.min.css` and `katex/katex.min.js`, so it works offline
and in the browser check. Same version as X_eTaL's `tools/katex`.
