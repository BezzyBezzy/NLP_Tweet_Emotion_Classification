# NLP Deep Learning: Classic Architectures & Transformer-Based Emotion Classification
**Authors:** Bezalel Spolter and Yair Shmueli

## Overview

This repository contains a comprehensive, two-part Natural Language Processing (NLP) deep learning project exploring text classification across different generations of neural network architectures.

* **Part 1: Classic Deep Learning & Word Embeddings** focuses on using recurrent structures (LSTMs and GRUs) paired with distinct word representation methods (GloVe and Word2Vec) to classify text sequences, complete with systematic hyperparameter random-search tuning.
* **Part 2: Emotion Classification of Tweets via Transformers** delivers a comparative study of modern pre-trained language models (BERTweet, Twitter-RoBERTa, and DeBERTa-v3) optimized for short-text social media sentiment analysis, followed by post-training compression evaluations (unstructured pruning, structural attention head pruning, and knowledge distillation).

The models were trained and evaluated on the [mteb/emotion](https://huggingface.co/datasets/mteb/emotion) dataset, using a standard 80/20 train/validation split. This dataset consists of English Twitter messages labeled with one of six discrete emotional categories: *sadness* (0), *joy* (1), *love* (2), *anger* (3), *fear* (4), or *surprise* (5).

---

## Part 1: Classic Deep Learning & Word Embeddings

### Methodology

1. **Data Preparation and Tokenization**
* Text inputs undergo standard case normalization (lowercasing).
* Sentences are tokenized into explicit word lists.
* An index-mapped vocabulary is built natively from the raw token distributions of the allocated datasets.


2. **Embeddings Evaluation**
* **GloVe**: Integrating static, pre-trained global vector spaces via standard downloads.
* **Word2Vec**: Training domain-specific continuous bag-of-words/skip-gram architectures natively on the target task dataset.


3. **Model Layout**
* Architectures utilize a single recurrent hidden layer, a standardized dropout regularization layer, and a terminal linear projection layer mapping to 6 raw outputs (logits).
* **Configuration A**: Long Short-Term Memory (LSTM) blocks layered on top of GloVe spaces.
* **Configuration B**: Bidirectional Gated Recurrent Units (GRU) layered on top of trained Word2Vec vectors.



### Hyperparameter Tuning Framework

To evaluate search optimization constraints, a random combination methodology was introduced to test 10 unique parameter variations against a calculated baseline. Due to structural target class imbalances, initial validation was cross-checked with a standard classification report (precision/recall balance) alongside temporal epoch validation tracking curves.

| Hyperparameter | Evaluated Options | Baseline Setting | Top GloVe + LSTM Config | Top Word2Vec + GRU Config |
| --- | --- | --- | --- | --- |
| **Batch Size** | 32, 64, 128 | 64 | 64 | 128 |
| **Learning Rate** | 0.001, 0.005, 0.01 | 0.001 | 0.001 | 0.005 |
| **Hidden Dimension Size** | 64, 128, 256 | 128 | 256 | 256 |
| **Dropout Rate** | 0.0, 0.3, 0.5 | 0.5 | 0.3 | 0.3 |

### Key Experimental Discoveries

* **Batch Regularization**: Fluctuations across batch parameters (32 to 128) demonstrated negligible distinct shifts in independent model performance boundaries.
* **Learning Curve Scaling**: The LSTM pipelines displayed stronger mathematical optimization tracking under tight, conservative learning bounds ($0.001$). Conversely, the GRU architectures favored high-step updates ($0.005$).
* **Hidden Dimension Scale**: Expanding structural hidden dimension matrices to 128 or 256 boosted validation performance over scaled 64-dimension baselines.
* **Baseline vs. Tuning Performance Summary**: The Word2Vec + GRU combination yielded higher top-tier bounds, achieving a peak accuracy score of **0.929** compared to the GloVe + LSTM peak of **0.925**, although this difference is slight.

---

## Part 2: Emotion Classification of Tweets via Transformers


### Abstract

This stage tracks and optimizes transformer performance constraints on social media texts using multi-class classification mappings across 6 target emotional profiles (*sadness, joy, love, anger, fear, surprise*). Phase A demonstrates that BERTweet, Twitter-RoBERTa, and DeBERTa-v3 achieve statistical parity, with all models establishing robust Macro F1 bounds $> 0.915$. Phase B evaluates structural compression options. Pruning matrices on Twitter-RoBERTa retained $>99\%$ performance retention. Knowledge Distillation into a miniature student model (`deberta-v3-xsmall`) achieved a Macro F1 score of **0.9125**, matching its larger teacher's execution with a fraction of the computational footprint.

### Data Preprocessing Pipeline

To control noisy anomalies characteristic of tweet syntax, an initial sanitization sequence was integrated prior to tokenization:

* Normalizing all missing/non-string properties to distinct empty string values.
* Standard case reduction (lowercasing).
* Syntactic contraction expansion (e.g., mapping `'Im'` directly to `'i am'`) to maximize lexical intersections with foundational pre-training dictionaries.
* Syllable elongation truncation (limiting repeated letter extensions beyond 2 elements, e.g., mapping `'gooooooood'` to `'good'`).
* Stripping arbitrary punctuation matrices and redundant spacing variables.
* *Note*: Exploratory evaluation verified that zero emoji tokens were natively present in this specific core raw data allocation.

### Phase A: Transformer Optimization & Benchmark Checks

Models were optimized with an initial conservative learning rate of $2\times10^{-5}$, initialized with a 500-step linear learning warmup sequence, a training batch size constraint of 32, a weight decay factor of $0.01$, and limited to 5 operational epochs.

Because of skewed class counts (e.g., *Joy* making up $\sim33.5\%$ of tokens vs. *Surprise* at just $\sim3.5\%$), **Macro Average F1-Score** was utilized as the primary baseline parameter rather than pure classification accuracy.

#### Quantitative Model Benchmarks

| Architecture Backbone | F1-Macro Score | Total Training Run Duration | Model Parameter Footprint |
| --- | --- | --- | --- |
| **Twitter-RoBERTa** (`twitter-roberta-base`) | **0.9164** | 30.7 minutes | **124M** |
| **BERTweet** (`bertweet-base`) | 0.9163 | 30.2 minutes | 135M |
| **DeBERTa-v3** (`deberta-v3-base`) | 0.9155 | **23.2 minutes** | 184M |

> **Statistical Significance Proofs:** Paired Sample T-Test metrics ($\alpha = 0.05$) cross-evaluated across all configuration pairs yielded $p$-values consistently above the critical significance floor ($p_{\text{RoBERTa vs. BERTweet}} = 0.9745$; $p_{\text{RoBERTa vs. DeBERTa}} = 0.8918$). These checks confirm that performance variances are statistically insignificant, making size and speed optimization the principal factors for deployment selection.

#### Loss & Prediction Error Analysis

* **Overfitting Horizons**: Loss evaluation curves demonstrated that validation loss functions flatlined after epoch 2, while training loss parameters continued to drop. This implies the start of overfitting cycles, proving that 3–4 training epochs are optimal for this configuration.
* **Semantic Overlap (The "Love" vs. "Joy" Frontier)**: Confusion matrix profiles revealed distinct target classification errors tracking along the *Love* and *Joy* semantic boundaries. Twitter-RoBERTa showed an operational bias toward *Love*, DeBERTa-v3 favored *Joy*, and BERTweet distributed its mistakes evenly between the two. This behavior highlights structural semantic overlap combined with severe dataset distribution bias (*Joy* at $33.5\%$ outperforming *Love* at $8.1\%$).

---

### Phase B: Post-Training Model Compression

Using our Phase A analysis, compression targets were prioritized based on physical size footprint (**Twitter-RoBERTa**) and training execution speed (**DeBERTa-v3**).

#### 1. Twitter-RoBERTa Pruning Pipelines

Following pruning applications, models were stabilized using 2 additional fine-tuning adjustment epochs.

* **Magnitude Pruning (Unstructured)**: Eliminating the bottom $10\%$ of global weights using a rigid weight magnitude threshold setting of $0.005$.
* **Head Pruning (Structured)**: Dynamically mapping and dropping the least informative attention head block per network layer. This removed 12 of the total 144 attention structures, targeting the lowest maximum attention weights.

#### 2. DeBERTa-v3 Knowledge Distillation

* **Teacher Core**: Optimized baseline DeBERTa-v3 configuration ($\text{F1} = 0.9155$).
* **Student Target**: `microsoft/deberta-v3-xsmall` (downscaling from 184M to 22M structural backbone parameters).
* **Loss Combination**: Balanced optimization target combining soft loss vectors (Teacher logits) with hard target parameters (Ground Truth), matching settings at $\alpha = 0.5$ and applying a Temperature scale of 4.

#### Compression Optimization Results

| Evaluation Run Framework | F1-Macro Score | Performance Delta | Total Architectural Footprint Shifts |
| --- | --- | --- | --- |
| **Baseline (Twitter-RoBERTa)** | 0.9164 | Reference | 124M Parameters (Dense) |
| **Magnitude Pruning (Unstructured)** | 0.9070 | -0.0094 | Sparse Matrix Topology |
| **Head Pruning (Structured)** | 0.9047 | -0.0117 | 132 Active Attention Heads remaining |
| **Baseline (DeBERTa-v3)** | 0.9155 | Reference | 184M Parameters |
| **Knowledge Distillation (Student)** | 0.9125 | **-0.0030** | **~88% Core Parameter Scale Drop (22M)** |

### Final Engineering Recommendations

While post-training compression alternatives prove that these deep architectures exhibit high over-parameterization density and can scale down with marginal performance drops, the full **Twitter-RoBERTa** model is recommended for ultimate production test configurations, as it secures the highest overall F1-Macro metric (**0.9164**).

---

## Bibliography

1. Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*.
2. Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L. T. (2020). Unified benchmark and comparative evaluation for tweet classification. *arXiv preprint arXiv:2010.12421*.
3. He, P., Gao, J., & Chen, W. (2021). Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing. *arXiv preprint arXiv:2111.09543*.
