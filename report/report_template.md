# xRFM: An Empirical and Interpretive Study

**Group:** [Group Name]
**Members:** [Name 1 (zID)], [Name 2 (zID)], ...

---

## 1. Introduction

Tabular data powers a large fraction of real-world machine-learning applications — electronic health records, financial transactions, sensor telemetry, census data — yet it remains the frontier where deep learning has struggled to dislodge gradient-boosted decision trees (GBDTs). Three structural properties of tabular data explain the gap: targets are often *non-smooth* and piecewise-constant along feature axes, feature scales and types are *heterogeneous*, and uninformative columns are common, all of which tree ensembles exploit and MLPs fight against [Grinsztajn et al., 2022]. Against this backdrop, kernel methods have been largely considered obsolete for tabular data because of (i) their fixed-geometry inductive bias and (ii) the O(n²) memory and O(n³) time cost of solving kernel ridge regression at scale.

**xRFM** [Beaglehole et al., 2025] addresses both limitations in one stroke. Its inner ingredient is the Recursive Feature Machine (RFM) [Radhakrishnan et al., 2024], which makes kernel machines *feature-learning* by iteratively reweighting inputs via the Average Gradient Outer Product (AGOP) matrix. The outer ingredient is a balanced binary tree whose internal splits are chosen along the *top AGOP eigenvector* of a locally-fitted RFM, yielding an O(n log n) training algorithm with O(log n) inference and — as a useful side-effect — per-leaf AGOPs that support subpopulation-specific interpretability.

We compare xRFM against XGBoost, Random Forest, and TabPFN-v2 on **five UCI datasets** (verified absent from both the TALENT benchmark and xRFM paper's meta-test suite) spanning regression, binary, and multiclass tasks, small to large n, and low to high d. Our headline takeaway is *xxx*. The report is structured as follows: §2 formalises AGOP and the xRFM algorithm; §3 reports test-set metrics, an interpretability comparison, and a scaling study; §4 connects empirical findings to theoretical predictions; §5 concludes.

## 2. Methodology

### 2.1 Average Gradient Outer Product (AGOP)

For a trained predictor $\hat f: \mathbb{R}^d \to \mathbb{R}$ and training set $\mathcal{S} = \{x^{(1)}, \ldots, x^{(n)}\}$, the AGOP is the empirical outer-product covariance of input gradients:

$$
\mathrm{AGOP}(\hat f, \mathcal{S}) \;=\; \tfrac{1}{n}\sum_{i=1}^{n} \nabla \hat f\big(x^{(i)}\big)\,\nabla \hat f\big(x^{(i)}\big)^\top \;\in\; \mathbb{R}^{d\times d}. \tag{1}
$$

**Geometric reading.** For any unit vector $v$, $v^\top M v = \mathbb{E}_x[(v \cdot \nabla \hat f(x))^2]$ is the expected squared directional derivative of $\hat f$ along $v$; so $M$ is the *covariance of sensitivities*. The diagonal $M_{ii} = \mathbb{E}[(\partial \hat f/\partial x_i)^2]$ measures per-coordinate sensitivity; the top eigenvectors span the *active subspace* [Constantine et al., 2014] — the directions carrying most variation in $\hat f$.

**Two interpretability strategies.**
- *Coordinate-wise*: read off $\{M_{ii}\}_i$ and rank features by marginal sensitivity.
- *Directional*: eigendecompose $M = V \Lambda V^\top$; the top eigenvector $v_1$ gives a *signed* linear combination — positive loadings contribute synergistically, negative loadings antagonistically. Fig. 7 of Beaglehole et al. (2025) shows this on Breast Cancer, where "concave points (mean)" (positive) and "compactness (error)" (negative) form the dominant axis of malignancy prediction.

**Comparison to other feature-importance methods.**

| Method | Supervised? | Interaction-aware? | Cost |
|---|---|---|---|
| AGOP diagonal | ✓ (via $\hat f$) | ✗ (marginal only) | $O(nd)$ after fit |
| AGOP eigenvectors | ✓ | ✓ (signed, joint) | $O(d^3)$ once |
| PCA loadings | ✗ (supervision-blind) | ✓ (unsigned, joint) | $O(d^3)$ |
| Mutual information | ✓ (direct on $y$) | ✗ (1-variable) | $O(n \log n)$ via density est. |
| Permutation importance | ✓ | ✗ | $O(d n)$ per model refit |

PCA computes $E[x x^\top]$ and is oblivious to the target; mutual information is model-free and marginal; permutation importance is black-box and unstable under feature correlation. AGOP occupies a unique point: supervised (through $\hat f$), joint (off-diagonals encode interactions), and single-pass (differentiable).

### 2.2 The xRFM Algorithm

xRFM nests an iterative kernel-ridge RFM inside the leaves of a balanced binary tree whose internal splits are chosen using an AGOP-based supervised criterion.

**Tree construction (recursive, Alg. A.2 of Beaglehole et al. 2025).** Given training set $\mathcal{D}$ with $|\mathcal{D}| > L_{\max}$: (i) subsample $N$ points; (ii) fit a one-iteration RFM split model $(K(X_s,X_s)+\lambda I)\alpha = y_s$; (iii) compute AGOP $M$; (iv) take the top eigenvector $v_1$ of $M$; (v) project all points onto $v_1$ and split at the median to ensure balance; (vi) recurse.

**Leaf training (Alg. A.1).** At each leaf, an RFM is fitted with the generalised kernel $K_{p,q}(x,x') = \exp(-\|x-x'\|_q^p / L^p)$ (Schoenberg 1942), where $q=2$ (rotation-invariant Laplace) or $q=p$ (axis-aligned product kernel). Starting with $M_0 = I$, iterate $\tau \leq 10$ times: (a) transform inputs by $M_t^{1/2}$; (b) solve ridge regression in the RKHS $(K+\lambda I)\alpha_t = y$; (c) compute new AGOP $M_{t+1}$; (d) normalise $M \leftarrow M/(\epsilon + \max_{ij}|M_{ij}|)$. Keep the iteration with best *validation* error.

**Inference.** A test point $x^*$ is routed through the tree (hard or soft thresholds) to its leaf $\ell$, then predicted as $\hat y = K(x^* \odot M_\ell^{1/2}, X_{M_\ell}) \alpha_\ell$. No ensembling: a single tree, one predictor per test point.

**Computational complexity.** Training is $O(n \log n)$ — leaves cap per-leaf cost at $O(L_{\max}^3)$ (constant in $n$), and median splits visit $O(n)$ points at each of $O(\log n)$ levels. Inference is $O(\log n + L_{\max})$. In contrast, a naive kernel method on the full $n$ costs $O(n^3)$ and $O(n^2)$ memory, becoming intractable above $\sim 70{,}000$ samples on a 40 GB GPU. xRFM preserves the feature-learning capability of RFM while bypassing its scaling wall.

### 2.3 Experimental Design

**Datasets (Table 1).** Five UCI datasets verified absent from the TALENT benchmark [Ye et al., 2024] and from xRFM's own evaluation suites. Two regression, three classification; one < 200, one > 300k; d ranges from 12 to 174; two mixed-type, three numeric-only.

| # | Dataset | UCI ID | n | d | Task | Features | Key property |
|---|---------|--------|---|---|------|---------|--------------|
| 1 | Seoul Bike Sharing Demand | 560 | 8,760 | 12 | Regression | 9 num + 3 cat | Strong temporal × weather interactions |
| 2 | Appliances Energy Prediction | 374 | 19,735 | 27 | Regression | All num | **Contains 2 ground-truth noise features (rv1, rv2)** |
| 3 | HCC Survival | 423 | 165 | 49 | Binary | 22 num + 27 cat | Small-n, heavy missingness |
| 4 | IDA2016 Scania APS Failures | 414 | 60,000 | 170 | Binary | All num | Class imbalance (~1.6% positive) |
| 5 | Crop Mapping (Winnipeg) | 525 | 50,000* | 174 | Multiclass (7) | All num | Satellite SAR + optical; *subsampled from 325,834 |

**Splits.** Stratified 60/20/20 train/val/test with `random_state=42` for reproducibility. Classification uses stratified splits; regression uses random.

**Preprocessing.** Median imputation for numerics, mode imputation for categoricals. For kernel / neural methods: standard-scaled numerics + one-hot encoded categoricals. For tree ensembles: numerics plus ordinal-encoded categoricals (no standardisation, preserving split structure).

**Hyperparameter tuning.** Optuna TPE sampler on the held-out validation split, with model-specific search spaces (xRFM: 25 trials; XGBoost: 40 trials; Random Forest: 20 trials; TabPFN: 1 trial — it has essentially no HPs). xRFM space matches Table A.1 of the xRFM paper: bandwidth $L \in [1, 200]$ log-uniform, ridge $\lambda \in [10^{-6}, 1]$ log-uniform, kernel exponent $p \in [0.7, 1.4]$, kernel type $\in \{K_{p,2}, K_{p,p}\}$, diagonal mode $\in \{\text{True}, \text{False}\}$.

**Metrics.** Regression: RMSE (primary) and $R^2$. Classification: Accuracy and AUC-ROC (macro-OvR for multiclass). Also: wall-clock training time and per-sample inference time.

**Compute.** All experiments run on Modal's A10G GPU instances with 32 GB RAM through a CUDA 12.4 Python 3.12 container. TabPFN-v2 is capped at 10,000 training samples (its published practical limit); datasets larger than this are stratified-subsampled before feeding.

## 3. Results

### 3.1 Main comparison

*[TABLE_PLACEHOLDER: main_table.md]*

**Summary of headline numbers.** [FILL after results]

### 3.2 Interpretability comparison on Appliances Energy

The Appliances Energy dataset [Candanedo et al., 2017] contains two engineered noise columns (`rv1` and `rv2`) introduced specifically to test whether a feature-importance method can correctly demote random variables. We extract per-leaf AGOP diagonals from a fitted xRFM, and compare against PCA-loaded magnitudes (unsupervised), mutual information scores (supervised, model-free), and permutation importance (supervised, black-box, using a tuned XGBoost as proxy model).

*[FIGURE_PLACEHOLDER: interpretability_appliances.png]*

Of particular interest is the rank given to the random features across methods:

*[TABLE_PLACEHOLDER: random_feature_ranks.csv]*

[FILL: narrative after results]

### 3.3 Scaling study on IDA2016

We train the four models on stratified subsamples of IDA2016's training partition ($n \in \{1k, 2.5k, 5k, 10k, 25k, 40k\}$) holding the test set fixed, and record both test AUC-ROC and wall-clock training time. TabPFN is omitted for $n > 10k$ as it exceeds its published scaling limit.

*[FIGURE_PLACEHOLDER: scaling_ida2016.png]*

[FILL: narrative after results]

## 4. Discussion

[FILL after results — see research brief for angles:
- Where xRFM should win / lose by theory
- AGOP vs MI disagreement points
- Per-leaf heterogeneity (if visible)
- Rotation-invariance vs axis-aligned patterns
- Comparison to TabPFN's in-context meta-learning approach
- Appliances random-feature test — which methods pass/fail?
]

## 5. Conclusion

[FILL: 0.3 pages. What we tested, what we found, what it tells us about xRFM's place in the tabular ML ecosystem.]

## References

1. Beaglehole, D., Holzmüller, D., Radhakrishnan, A., & Belkin, M. (2025). *xRFM: Accurate, scalable, and interpretable feature learning models for tabular data*. arXiv:2508.10053.
2. Radhakrishnan, A., Beaglehole, D., Pandit, P., & Belkin, M. (2024). Mechanism for feature learning in neural networks and backpropagation-free machine learning models. *Science*, **383**(6690).
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *KDD 2016*.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, **45**(1), 5–32.
5. Hollmann, N., Müller, S., Purucker, L., et al. (2025). Accurate predictions on small data with a tabular foundation model. *Nature*, **637**.
6. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *NeurIPS 2022 Datasets & Benchmarks Track*.
7. Ye, H.-J., Liu, S.-Y., et al. (2024). A Closer Look at Deep Learning on Tabular Data (TALENT). arXiv:2407.00956.
8. Constantine, P. G., Dow, E., & Wang, Q. (2014). Active subspace methods in theory and practice. *SIAM J. Sci. Comput.*, **36**(4).
9. Beaglehole, D., Súkeník, P., Mondelli, M., & Belkin, M. (2024). Average gradient outer product as a mechanism for deep neural collapse. In *NeurIPS 2024*.
10. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
11. Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, **140**, 81–97.
12. Dua, D., & Graff, C. (2017). UCI Machine Learning Repository. University of California, Irvine.
