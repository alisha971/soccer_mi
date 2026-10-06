# Why SEL-4L's attention heads do not carry the identity–context decision: audit and literature

*Written after MB14–MB18 (validation only). Each statement below is either a measured project result (with the notebook that measured it) or a published finding (with its source). Hypotheses are labelled as such.*

## 1. What we actually measured

| Evidence | Number | Source |
|---|---|---|
| All 8 heads alone (MLPs removed) reproduce | ~1% of Δ, identity effect and context effect | MB14 |
| All 4 MLPs alone (heads removed at every position) reproduce | 97% of Δ, 91% of the identity effect, 96% of the context effect | MB14 |
| Same, on the other 4 SEL-4L seeds and on REF-2L / REF-4L | MLP-only patching profile, attention outputs recover ≤ 0.08 | MB18 |
| Swapping the 15 earlier events (scored event kept) | moves Δ by 9% of what swapping the scored event's context does | MB15 Q2 |
| Single-head ablation, NLL on **all** scored validation events | +0.002 to +0.034 nats | MB07 |
| Where heads look (excess over uniform attention) | own position +0.15 to +0.29; previous event +0.13 to +0.18 (layer 0) | MB07 |
| Test NLL: logistic regression on the current event | 1.0649 | v3 Table 8 |
| Test NLL: MLP on current + 3 previous events | 0.9559 | v3 Table 8 |
| Test NLL: SEL-4L (attention over 16 events) | 0.9148 (0.041 better than the MLP) | v3 Table 8 |
| Context length in the capacity benchmark | 16-event windows best at every depth | v3 Section 5 |

**Reading.** Attention is not useless for the model as a whole: across all events, longer windows help and a 16-event Transformer beats a 3-event MLP by 0.04 nats. But on **conflict events** attention does essentially nothing, and the decision is made from the scored event's own inputs by a serial chain of MLPs.

## 2. Why: four reasons, from strongest evidence to weakest

### 2.1 The data: each event token already summarises the previous event (strong evidence)

Every token contains, besides the acting player, team, role and competition, the previous event's action, outcome and end zone, whether the same player acted before, the time since the previous event, the possession state and the possession length. Most of what attention would fetch from position t−1 is therefore **already in token t**. Measured consequences:
- In MB13 and MB15 Q4, the context inputs that move Δ most are exactly these summaries: same player as previous, possession change, previous action, time gap, previous outcome.
- The earlier events add little (9%, MB15 Q2).
- A model that sees the current event plus three previous events (B4) is only 0.041 nats behind the Transformer.

The sequence is close to first-order Markov once the previous event is summarised in the token. A Transformer on such data has little left for attention to do.

### 2.2 The conflict set selects events decided by the current token (strong evidence, by construction)

A conflict is an event where the identity reference and the **context reference** disagree strongly. The context reference sees only the current event's own fields (its inputs are listed in MB01). On conflict rows it is very confident: median probability of its top action 0.96, median gap to the second action 0.94 (MB01 10A.7). So conflict events are, by construction, events where the current token alone almost fixes the action. On these events the MLPs reading that token are expected to dominate. Any result "attention is not involved **in arbitration**" is partly a result of how the population was chosen.

**Hypothesis to test:** attention matters more on events the context reference is *unsure* about. MB07's NLL numbers over all events point that way, but it is not tested directly (see Section 5).

### 2.3 The input design: identity and context enter at the same position, as a sum (strong evidence)

Each token is the **sum** of its factor embeddings. Player and context for the scored event are both in the scored token from the first layer on. No information has to be moved between positions to bring "who" and "what situation" together. Combining them is a per-position, non-linear operation: exactly what an MLP block does. MB14's Shapley values even show the order: early MLPs lean slightly towards identity, late ones towards context. MB18 shows the MLPs work in series (removing L0MLP costs 7.6 nats in total, but only 0.57 directly).

### 2.4 The architecture: without attention, a Transformer *is* a residual MLP (definition)

A Transformer block is attention (mixes positions) followed by an MLP (acts on each position separately), both added to a residual stream. If attention contributes nothing at the scored position, the model on that position reduces to embedding → 4 residual MLP blocks with LayerNorm → output. That is the "ResNet-like MLP" that the tabular deep-learning literature finds to be a strong baseline (Gorishniy et al., 2021). So "the Transformer behaves like a plain neural network" is not a malfunction. It is the model using the part of its architecture the data needs.

**Probably not the reason, but untested here:** too few heads. SEL-4L has 2 heads per layer, and the MI notebooks never varied this. On football events, Seq2Event found that Transformers with more than 2 heads did worse, not better (Simpson et al., 2022).

## 3. What the literature says

### 3.1 Deep learning, Transformers and tabular data

- **Gorishniy et al. (2021), *Revisiting Deep Learning Models for Tabular Data*, NeurIPS.** A ResNet-like MLP is a strong baseline. Their FT-Transformer (attention **across features**, not across time) wins on most tasks, but no model is universally best against gradient-boosted trees. Relevance: our token sums the features, so feature interactions must be learned by the MLPs, the role FT-Transformer gives to attention. [paper](https://arxiv.org/abs/2106.11959)
- **Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS Datasets & Benchmarks.** Tree models remain stronger on medium-sized tabular data; neural networks are hurt by uninformative features and irregular target functions. Relevance: player and team inputs behave like weakly informative features (removing them improves NLL by about 0.02 nats in the v3 ablations). [paper](https://arxiv.org/abs/2207.08815)
- **Shwartz-Ziv & Armon (2022), *Tabular data: Deep learning is not all you need*, Information Fusion.** XGBoost outperformed the deep tabular models they tested across datasets. [paper](https://arxiv.org/abs/2106.03253)
- **Huang et al. (2020), *TabTransformer*.** Attention over categorical feature embeddings, to make each feature's embedding depend on the others. It shows what attention is for in tabular Transformers: relating features. Our design does not ask attention to do that. [paper](https://arxiv.org/abs/2012.06678)
- **Hollmann et al. (2023), *TabPFN*, ICLR.** A Transformer pre-trained on synthetic data that does in-context learning on small tabular tasks; attention there runs over training examples. Different use of attention, cited only to separate the meanings of "Transformer for tabular data". [paper](https://arxiv.org/abs/2207.01848)

### 3.2 Sequences where attention adds little

- **Zeng et al. (2023), *Are Transformers Effective for Time Series Forecasting?*, AAAI.** Simple one-layer linear models beat Transformer forecasters on standard benchmarks. Order information that attention discards, or does not need, limits what attention adds. [paper](https://doi.org/10.1609/aaai.v37i9.26317)
- **Bosser & Ben Taieb (2023), *On the Predictive Accuracy of Neural Temporal Point Process Models for Continuous-time Event Data*, TMLR.** In marked event sequences, the history encoder (GRU, self-attention, or a constant) matters much less than expected: "relying solely on a subset of the most recent observed occurrences can yield comparable performance to encoding the entire historical context." This matches MB15 Q2 closely. [paper](https://arxiv.org/abs/2306.17066)
- **Liu et al. (2021), *Pay Attention to MLPs* (gMLP), NeurIPS.** MLP-based models with gating can match Transformers on several tasks, so self-attention is not always required. [paper](https://arxiv.org/abs/2105.08050)

### 3.3 Football and sports event models

- **Simpson et al. (2022), *Seq2Event*, KDD.** Next-event prediction on football event data. An LSTM (test loss 0.332) beat the Transformer (0.362); Transformers with more than 2 heads did worse; very short sequences (5 events) did poorly. History matters somewhat on football events, but attention is not the best way to use it there. Their tokens carry fewer engineered summaries than ours. [paper](https://eprints.soton.ac.uk/458099/)
- **Mendes-Neves, Meireles & Mendes-Moreira (2024), *Towards a Foundation Large Events Model for Soccer*, Machine Learning.** Their Large Events Model conditions on **one** previous event. The follow-up, *Forecasting Events in Soccer Matches Through Language* (2024), with three previous events improved event-type accuracy by 6.5%. A little extra history helps; most of the signal is in the most recent event. [LEM](https://arxiv.org/abs/2402.06815) · [follow-up](https://arxiv.org/abs/2402.06820)
- **Alcorn & Nguyen (2021), *baller2vec*.** A multi-entity Transformer on basketball **tracking** data with learned player embeddings. The embeddings came to encode player attributes, and several heads took on basketball functions (for example anticipating passes). Contrast: when tokens are separate players who interact at the same moment, attention has a natural job. In our event stream one token is one action, and the previous action is already summarised in it. [paper](https://arxiv.org/abs/2102.03291)

### 3.4 Attention heads are often dispensable; MLPs store associations

- **Michel, Levy & Neubig (2019), *Are Sixteen Heads Really Better than One?*, NeurIPS.** Many heads can be removed at test time with little loss; some layers can be reduced to a single head. [paper](https://arxiv.org/abs/1905.10650)
- **Voita et al. (2019), *Analyzing Multi-Head Self-Attention*, ACL.** A few specialised heads do most of the work; most can be pruned with little loss. [paper](https://arxiv.org/abs/1905.09418)
- **Geva et al. (2021), *Transformer Feed-Forward Layers Are Key-Value Memories*, EMNLP.** MLP layers act as key–value memories that map input patterns to output distributions. Relevance: our MLP chain maps (role, previous action, possession change, coupling) to an action preference. In MB16 those player-blind labels alone explain 89% of Δ through the MLPs. [paper](https://aclanthology.org/2021.emnlp-main.446)
- **Meng et al. (2022), *Locating and Editing Factual Associations in GPT* (ROME), NeurIPS.** Associations about a subject are recalled in MLPs at the subject's token position. Analogue (hypothesis): what is "known" about the acting player would be read out by MLPs at the scored event's own position, as MB06 and MB14 find. [paper](https://arxiv.org/abs/2202.05262)

### 3.5 Interpreting the MI results

- **McGrath et al. (2023), *The Hydra Effect*,** and **Rushing & Nanda (2024), *Explorations of Self-Repair in Language Models*, ICML.** When components are ablated, later components often compensate. MB18 finds the opposite in SEL-4L: removing an early MLP costs much **more** than its direct effect, because later MLPs build on it (amplification, not repair). [Hydra](https://arxiv.org/abs/2307.15771) · [self-repair](https://arxiv.org/abs/2402.15390)
- **Heimersheim & Nanda (2024), *How to use and interpret activation patching*.** Guidance on denoising vs noising and on metrics. Relevant to the MB16 correction: sign-aligned ratios behave badly when the denominator is small and noisy. [paper](https://arxiv.org/abs/2404.15255)
- **Geirhos et al. (2020), *Shortcut learning in deep neural networks*, Nature Machine Intelligence.** Networks use the easiest predictive features. On conflicts, the engineered previous-event summaries are such features. [paper](https://arxiv.org/abs/2004.07780)

## 4. What the paper can say

> On conflict events, SEL-4L arbitrates between identity and context with a serial chain of MLPs at the scored event, not with attention. Each event token already summarises the previous event, and conflict events are by construction those that the current event's context fields nearly decide. On this population the model therefore acts as a per-event residual MLP. Over all events, attention still adds a small amount (0.04 nats over an MLP with three previous events), in line with findings that recent history carries most of the signal in event sequences (Bosser & Ben Taieb, 2023) and that a few heads often suffice (Michel et al., 2019; Voita et al., 2019).

Do **not** write "attention is not used by the model" without the qualifier "on conflict events".

## 5. Open checks (not run)

1. **All heads removed, NLL on every validation event,** split by how confident the context reference is. This tests whether attention matters exactly where the current token is ambiguous.
2. **How much of the previous event's raw fields can be decoded from the current token's embedding alone?** This quantifies the redundancy argued in 2.1.
3. **Head importance on non-conflict events where the previous event is a different team's action, or follows a stoppage.** Long-range context (possession build-up) is where attention would be expected to help.

## 6. Addendum: Seq2Event's "LSTM beats Transformer", read closely

- **Setup** (Simpson et al., KDD 2022):
  - Wyscout open data; 138 matches (69 train, 7 validation, 62 test).
  - Each event has 1 categorical action (pass, dribble, cross, shot) and 10 continuous features, including the changes since the previous event (Δx, Δy, Δtime, distance since previous).
  - The target is the next event's action and location.
  - Loss = 5 × class-weighted cross-entropy + RMSE of the location.
  - 145 models were fitted in all.
- **Result:**
  - Best LSTM (sequence length 100, 1 layer, hidden size 8): test loss 0.332.
  - Best Transformer (sequence length 40, 1 layer, hidden size 8, 1 head): 0.362. The paper also quotes 0.379 for an "equivalent configuration".
  - No seeds, intervals or significance tests are reported, and the two best models used different sequence lengths.
- **The authors' explanation** is one sentence: "it has been observed that on more complex sequences, LSTM may provide better fit due to controlling the exposure of the memory to the network". The citation, Cahuantzi, Chen & Güttel (2021), compares LSTMs with GRUs on symbolic sequences and contains no Transformer.
- **Reading:** a single-run gap of 0.03 between very small models, compared at different sequence lengths and with no uncertainty, is weak evidence that LSTMs suit football event data better. It is consistent with general findings that Transformers are less sample-efficient than recurrent models on small data:
  - Huang et al., ICLR 2023, *Encoding Recurrence into Transformers*.
  - *On the "Induction Bias" in Sequence Models*, arXiv 2602.18333.
  - Tran, Bisazza & Monz, EMNLP 2018.
  
  But the paper itself does not test this.
