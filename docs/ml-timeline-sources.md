# ML timeline: sources

Where each event in `timeline/data/timeline.json` comes from: a row
for every event but the demos (which are this repository's own), with
its year, its month (or `-` when only the year is certain) and its
title exactly as the event writes them, what was checked (the authors,
the full title, the venue and the date), and the link it was checked
at, which is where the event leads. The core's test
`every_event_has_its_source` holds the two together: an event without
its row, a row without its event, a row twice, or a link that is not
the event's fails it.

How a date is chosen:

- A paper on arXiv stands at the month of its first version (from the
  arXiv record's submission history); its later venue is noted.
- A journal paper stands at its issue's month, or at its online date
  when that comes first (AlphaGo Zero, AlphaFold).
- A conference paper stands at the conference's month; a report or a
  thesis at the month on its title page, when it has one.
- A book, a memo or a machine whose month is not reliably known has
  only its year, and stands mid-year. Lloyd's k-means stands at its
  1957 memo, with its 1982 publication linked.
- A milestone stands at the month of the event or the announcement.

Every link was opened and found to lead to the work it names; DOIs
were checked against their Crossref records. Pictures are this
repository's own screenshots only.

| Year | Month | Title | Checked | Link |
| ---- | ----- | ----- | ------- | ---- |
| 1943 | 12 | A logical calculus of the ideas immanent in nervous activity | McCulloch and Pitts, A logical calculus of the ideas immanent in nervous activity, Bulletin of Mathematical Biophysics 5 | <https://doi.org/10.1007/BF02478259> |
| 1949 | - | The Organization of Behavior | Hebb, The Organization of Behavior: A Neuropsychological Theory (Wiley, 1949) | <https://en.wikipedia.org/wiki/Organization_of_Behavior> |
| 1950 | 10 | Computing Machinery and Intelligence | Turing, Computing Machinery and Intelligence, Mind 59 (236) | <https://doi.org/10.1093/mind/LIX.236.433> |
| 1951 | - | SNARC | Minsky and Edmonds, the Stochastic Neural Analog Reinforcement Calculator, Princeton, summer 1951 | <https://en.wikipedia.org/wiki/Stochastic_Neural_Analog_Reinforcement_Calculator> |
| 1952 | - | Samuel's checkers program | Samuel's checkers on the IBM 701, 1952 (also https://en.wikipedia.org/wiki/IBM_701) | <https://en.wikipedia.org/wiki/Arthur_Samuel_(computer_scientist)> |
| 1956 | - | The Dartmouth workshop | The Dartmouth workshop (summer 1956) | <https://en.wikipedia.org/wiki/Dartmouth_workshop> |
| 1957 | - | Lloyd's algorithm | Lloyd, Least squares quantization in PCM, IEEE Trans. Information Theory 28 (2), March 1982; the 1957 memo per https://en.wikipedia.org/wiki/K-means_clustering | <https://doi.org/10.1109/TIT.1982.1056489> |
| 1958 | 11 | The perceptron | Rosenblatt, The perceptron, Psychological Review 65 (6) | <https://doi.org/10.1037/h0042519> |
| 1959 | 7 | Some studies in machine learning using the game of checkers | Samuel, Some Studies in Machine Learning Using the Game of Checkers, IBM J. Res. Dev. 3 (3), July 1959 | <https://doi.org/10.1147/rd.33.0210> |
| 1960 | 8 | Adaptive switching circuits | Widrow and Hoff, Adaptive Switching Circuits, IRE WESCON Convention Record part 4, August 1960 (also https://isl.stanford.edu/~widrow/publications.html) | <https://en.wikipedia.org/wiki/ADALINE> |
| 1965 | - | Cybernetic predicting devices | Ivakhnenko and Lapa, Cybernetic Predicting Devices (1965), as cited by Schmidhuber; GMDH named in 1968 | <https://en.wikipedia.org/wiki/Group_method_of_data_handling> |
| 1967 | 1 | Nearest neighbor pattern classification | Cover and Hart, Nearest neighbor pattern classification, IEEE Trans. Information Theory 13 (1), January 1967 | <https://doi.org/10.1109/TIT.1967.1053964> |
| 1967 | - | Some methods for classification and analysis of multivariate observations | MacQueen, Proc. Fifth Berkeley Symposium on Mathematical Statistics and Probability, vol. 1, 1967 | <https://projecteuclid.org/euclid.bsmsp/1200512992> |
| 1969 | - | Perceptrons | Minsky and Papert, Perceptrons (MIT Press) | <https://en.wikipedia.org/wiki/Perceptrons_(book)> |
| 1970 | - | Reverse-mode differentiation | Linnainmaa, master's thesis, University of Helsinki, 1970; Taylor expansion of the accumulated rounding error, BIT 16 (2), June 1976, https://doi.org/10.1007/BF01931367 | <https://en.wikipedia.org/wiki/Seppo_Linnainmaa> |
| 1973 | - | The Lighthill report | Lighthill, Artificial Intelligence: A General Survey, Science Research Council, 1973 | <https://en.wikipedia.org/wiki/Lighthill_report> |
| 1974 | 8 | Beyond regression | Werbos, Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences, PhD thesis, Harvard, August 1974 | <https://en.wikipedia.org/wiki/Paul_Werbos> |
| 1980 | 4 | Neocognitron | Fukushima, Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position, Biological Cybernetics 36 (4), April 1980 | <https://doi.org/10.1007/BF00344251> |
| 1982 | 4 | Neural networks and physical systems with emergent collective computational abilities | Hopfield, PNAS 79 (8), April 1982 | <https://doi.org/10.1073/pnas.79.8.2554> |
| 1986 | 10 | Learning representations by back-propagating errors | Rumelhart, Hinton and Williams, Learning representations by back-propagating errors, Nature 323 | <https://doi.org/10.1038/323533a0> |
| 1989 | 12 | Backpropagation applied to handwritten zip code recognition | LeCun et al., Backpropagation applied to handwritten zip code recognition, Neural Computation 1 (4) | <https://doi.org/10.1162/neco.1989.1.4.541> |
| 1990 | 3 | Word association norms, mutual information, and lexicography | Church and Hanks, Computational Linguistics 16 (1), March 1990 | <https://aclanthology.org/J90-1003/> |
| 1991 | 2 | Adaptive mixtures of local experts | Jacobs, Jordan, Nowlan and Hinton, Neural Computation 3 (1), February 1991 | <https://doi.org/10.1162/neco.1991.3.1.79> |
| 1992 | 5 | TD-Gammon | Tesauro, Practical issues in temporal difference learning, Machine Learning 8 (3-4), May 1992 | <https://doi.org/10.1007/BF00992697> |
| 1995 | 9 | Support-vector networks | Cortes and Vapnik, Machine Learning 20 (3), September 1995 | <https://doi.org/10.1007/BF00994018> |
| 1997 | 5 | Deep Blue beats Kasparov | Deep Blue beats Kasparov | <https://en.wikipedia.org/wiki/Deep_Blue_versus_Garry_Kasparov> |
| 1997 | 11 | Long short-term memory | Hochreiter and Schmidhuber, Long short-term memory, Neural Computation 9 (8) | <https://doi.org/10.1162/neco.1997.9.8.1735> |
| 1998 | 11 | Gradient-based learning applied to document recognition | LeCun, Bottou, Bengio and Haffner, Proc. IEEE 86 (11), November 1998 | <https://doi.org/10.1109/5.726791> |
| 2001 | 10 | Random forests | Breiman, Random Forests, Machine Learning 45 (1), October 2001 | <https://doi.org/10.1023/A:1010933404324> |
| 2006 | 7 | A fast learning algorithm for deep belief nets | Hinton, Osindero and Teh, Neural Computation 18 (7), July 2006 | <https://doi.org/10.1162/neco.2006.18.7.1527> |
| 2009 | 6 | ImageNet: a large-scale hierarchical image database | Deng et al. (6 authors), CVPR 2009, June 2009 | <https://doi.org/10.1109/CVPR.2009.5206848> |
| 2010 | 6 | Rectified linear units improve restricted Boltzmann machines | Nair and Hinton, ICML 2010, June 2010 | <https://icml.cc/Conferences/2010/papers/432.pdf> |
| 2011 | 2 | Watson wins Jeopardy! | IBM Watson's Jeopardy! matches, aired February 14-16, 2011 | <https://en.wikipedia.org/wiki/Watson_(computer)> |
| 2012 | 12 | ImageNet classification with deep convolutional neural networks | Krizhevsky, Sutskever and Hinton, ImageNet classification with deep convolutional neural networks, NeurIPS 2012 (the DOI is its 2017 reprint in Communications of the ACM) | <https://doi.org/10.1145/3065386> |
| 2013 | 1 | Efficient estimation of word representations in vector space | Mikolov, Chen, Corrado and Dean, arXiv v1 2013-01-16; ICLR 2013 workshop | <https://arxiv.org/abs/1301.3781> |
| 2013 | 12 | Auto-encoding variational Bayes | Kingma and Welling, arXiv v1 2013-12-20; ICLR 2014 | <https://arxiv.org/abs/1312.6114> |
| 2014 | 6 | Generative adversarial networks | Goodfellow et al. (8 authors), arXiv v1 2014-06-10; NIPS 2014 as Generative Adversarial Nets | <https://arxiv.org/abs/1406.2661> |
| 2014 | 6 | Dropout: a simple way to prevent neural networks from overfitting | Srivastava et al. (5 authors), JMLR 15 (56), published June 2014 (the PDF's header) | <https://jmlr.org/papers/v15/srivastava14a.html> |
| 2014 | 9 | Neural machine translation by jointly learning to align and translate | Bahdanau, Cho and Bengio, arXiv v1 2014-09-01; ICLR 2015 | <https://arxiv.org/abs/1409.0473> |
| 2014 | 12 | Adam: a method for stochastic optimization | Kingma and Ba, Adam: a method for stochastic optimization | <https://arxiv.org/abs/1412.6980> |
| 2015 | 2 | Batch normalization | Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, arXiv v1 2015-02-11; ICML 2015 | <https://arxiv.org/abs/1502.03167> |
| 2015 | 3 | Deep unsupervised learning using nonequilibrium thermodynamics | Sohl-Dickstein, Weiss, Maheswaranathan and Ganguli, arXiv v1 2015-03-12; ICML 2015 | <https://arxiv.org/abs/1503.03585> |
| 2015 | 12 | Deep residual learning for image recognition | He, Zhang, Ren and Sun, arXiv v1 2015-12-10; CVPR 2016 | <https://arxiv.org/abs/1512.03385> |
| 2016 | 3 | AlphaGo beats Lee Sedol | AlphaGo beats Lee Sedol | <https://en.wikipedia.org/wiki/AlphaGo_versus_Lee_Sedol> |
| 2016 | 5 | Ternary weight networks | Fengfu Li and Bin Liu, arXiv v1 2016-05-16 (later versions add authors; ICASSP 2023) | <https://arxiv.org/abs/1605.04711> |
| 2017 | 1 | Outrageously large neural networks: the sparsely-gated mixture-of-experts layer | Shazeer et al. (7 authors), arXiv v1 2017-01-23; ICLR 2017 | <https://arxiv.org/abs/1701.06538> |
| 2017 | 6 | Attention is all you need | Vaswani et al., Attention is all you need | <https://arxiv.org/abs/1706.03762> |
| 2017 | 7 | Proximal policy optimization algorithms | Schulman et al. (5 authors), arXiv v1 2017-07-20 | <https://arxiv.org/abs/1707.06347> |
| 2017 | 10 | AlphaGo Zero | Silver et al. (17 authors), Mastering the game of Go without human knowledge, Nature 550, online 2017-10-19 | <https://doi.org/10.1038/nature24270> |
| 2018 | 3 | World models | Ha and Schmidhuber, arXiv v1 2018-03-27; NeurIPS 2018 as Recurrent World Models Facilitate Policy Evolution | <https://arxiv.org/abs/1803.10122> |
| 2018 | 6 | Improving language understanding by generative pre-training | Radford, Narasimhan, Salimans and Sutskever, OpenAI report, June 2018 (the PDF's date) | <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf> |
| 2018 | 10 | BERT | Devlin et al., BERT | <https://arxiv.org/abs/1810.04805> |
| 2019 | 2 | Language models are unsupervised multitask learners | Radford et al. (6 authors), OpenAI report, February 2019 (the PDF's date) | <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf> |
| 2020 | 1 | Scaling laws for neural language models | Kaplan et al. (10 authors), arXiv v1 2020-01-23 | <https://arxiv.org/abs/2001.08361> |
| 2020 | 2 | GLU variants improve Transformer | Shazeer, arXiv v1 2020-02-12 | <https://arxiv.org/abs/2002.05202> |
| 2020 | 5 | Language models are few-shot learners | Brown et al., Language models are few-shot learners (GPT-3) | <https://arxiv.org/abs/2005.14165> |
| 2020 | 6 | Denoising diffusion probabilistic models | Ho, Jain and Abbeel, Denoising diffusion probabilistic models | <https://arxiv.org/abs/2006.11239> |
| 2020 | 10 | An image is worth 16x16 words | Dosovitskiy et al. (12 authors), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, arXiv v1 2020-10-22; ICLR 2021 | <https://arxiv.org/abs/2010.11929> |
| 2021 | 1 | Switch Transformers | Fedus, Zoph and Shazeer, Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, arXiv v1 2021-01-11; JMLR 23 (120), 2022 | <https://arxiv.org/abs/2101.03961> |
| 2021 | 2 | Learning transferable visual models from natural language supervision | Radford et al. (12 authors), arXiv v1 2021-02-26; ICML 2021 | <https://arxiv.org/abs/2103.00020> |
| 2021 | 6 | LoRA: low-rank adaptation of large language models | Hu et al. (8 authors), arXiv v1 2021-06-17; ICLR 2022 | <https://arxiv.org/abs/2106.09685> |
| 2021 | 7 | Highly accurate protein structure prediction with AlphaFold | Jumper et al. (34 authors), Nature 596, online 2021-07-15 | <https://doi.org/10.1038/s41586-021-03819-2> |
| 2021 | 12 | High-resolution image synthesis with latent diffusion models | Rombach et al. (5 authors), arXiv v1 2021-12-20; CVPR 2022 | <https://arxiv.org/abs/2112.10752> |
| 2022 | 3 | Training language models to follow instructions with human feedback | Ouyang et al. (20 authors), arXiv v1 2022-03-04; NeurIPS 2022 | <https://arxiv.org/abs/2203.02155> |
| 2022 | 11 | ChatGPT | ChatGPT released | <https://en.wikipedia.org/wiki/ChatGPT> |
| 2023 | 2 | LLaMA: open and efficient foundation language models | Touvron et al. (14 authors), arXiv v1 2023-02-27 | <https://arxiv.org/abs/2302.13971> |
| 2023 | 5 | QLoRA | Dettmers, Pagnoni, Holtzman and Zettlemoyer, QLoRA: Efficient Finetuning of Quantized LLMs, arXiv v1 2023-05-23; NeurIPS 2023 | <https://arxiv.org/abs/2305.14314> |
| 2023 | 5 | Direct preference optimization | Rafailov et al. (6 authors), Direct Preference Optimization: Your Language Model is Secretly a Reward Model, arXiv v1 2023-05-29; NeurIPS 2023 | <https://arxiv.org/abs/2305.18290> |
| 2023 | 8 | LoRA-FA | Zhang et al. (5 authors), LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning, arXiv v1 2023-08-07 (retitled in v3, May 2026) | <https://arxiv.org/abs/2308.03303> |
| 2023 | 10 | VeRA: vector-based random matrix adaptation | Kopiczko, Blankevoort and Asano, arXiv v1 2023-10-17; ICLR 2024 | <https://arxiv.org/abs/2310.11454> |
| 2023 | 10 | BitNet: scaling 1-bit Transformers for large language models | Wang et al. (10 authors), arXiv v1 2023-10-17 | <https://arxiv.org/abs/2310.11453> |
| 2023 | 12 | Mixtral 8x7B | Mistral AI's announcement, December 11, 2023 | <https://mistral.ai/news/mixtral-of-experts> |
| 2024 | 1 | Mixtral of experts | Jiang et al. (26 authors), arXiv v1 2024-01-08 | <https://arxiv.org/abs/2401.04088> |
| 2024 | 2 | The era of 1-bit LLMs | Ma et al. (10 authors), The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits, arXiv v1 2024-02-27 | <https://arxiv.org/abs/2402.17764> |
| 2024 | 10 | The Nobel Prize in Physics | The Nobel Prize in Physics 2024, announced October 8, 2024 (the press release) | <https://www.nobelprize.org/prizes/physics/2024/summary/> |
| 2024 | 10 | The Nobel Prize in Chemistry | The Nobel Prize in Chemistry 2024, announced October 9, 2024 (the press release) | <https://www.nobelprize.org/prizes/chemistry/2024/summary/> |
| 2024 | 12 | DeepSeek-V3 technical report | DeepSeek-AI, arXiv v1 2024-12-27 | <https://arxiv.org/abs/2412.19437> |
| 2025 | 1 | DeepSeek-R1 | DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, arXiv v1 2025-01-22; Nature 645 (2025), https://doi.org/10.1038/s41586-025-09422-z | <https://arxiv.org/abs/2501.12948> |
| 2025 | 3 | The Turing Award for reinforcement learning | ACM's announcement of the 2024 A.M. Turing Award, March 5, 2025 | <https://www.acm.org/media-center/2025/march/turing-award-2024> |
| 2025 | 7 | Gold at the Mathematical Olympiad | Google DeepMind's blog, July 21, 2025 | <https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/> |
| 2026 | 9 | X_eTaL | X_eTaL's repository, created September 27, 2026 (GitHub, and its first commit) | <https://github.com/softwarewrighter/X_eTaL> |
| 2026 | 10 | X_eTaL v0.1.0 | X_eTaL's tag v0.1.0, October 5, 2026 | <https://github.com/softwarewrighter/X_eTaL/tree/v0.1.0> |
| 2026 | 10 | X_eTaL-ML | This repository, created October 3, 2026 (GitHub, and its first commit) | <https://github.com/softwarewrighter/X_eTaL-ML> |
