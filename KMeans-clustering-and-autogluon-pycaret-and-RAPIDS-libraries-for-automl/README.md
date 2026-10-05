# Data Mining: K-means, AutoML and GPU Data Science

A set of Jupyter notebooks for SJSU Data Mining. They start with K-means clustering. They then cover three tools that speed up the machine learning workflow: PyCaret, AutoGluon and NVIDIA RAPIDS. Each tool has a long "Zero to Hero" notebook that builds up from the basics. PyCaret and AutoGluon also have a shorter "Capabilities Tour" notebook.

Every notebook is organised into numbered Parts and Tasks. Each Task explains an idea, runs the code and ends with a short summary.

## Video walkthroughs

Each Zero to Hero notebook has a recorded walkthrough that explains it. AutoGluon and PyCaret are each split into two parts.

| Notebook | Video |
|---|---|
| K-means Zero to Hero | [Watch](https://youtu.be/6GO0iMu7rpc) |
| AutoGluon Zero to Hero, part 1 | [Watch](https://youtu.be/xn1zDssvsGc) |
| AutoGluon Zero to Hero, part 2 | [Watch](https://youtu.be/fhUJqic2kmA) |
| NVIDIA RAPIDS Zero to Hero | [Watch](https://youtu.be/Svu8EvO1Zgc) |
| PyCaret Zero to Hero, part 1 | [Watch](https://youtu.be/OvIlHmWX9k4) |
| PyCaret Zero to Hero, part 2 | Coming soon |

## Notebooks

### [final_kmeans_zero_to_hero.ipynb](final_kmeans_zero_to_hero.ipynb)
A full course on K-means clustering in 29 tasks. It covers:
- What a cluster is, and Lloyd's algorithm written from scratch.
- Initialisation: the random-start trap, K-means++, and handling empty clusters.
- Bisecting K-means.
- Variants: K-medians, K-medoids, Mini-Batch, fuzzy c-means, Gaussian mixtures and kernel K-means.
- Cluster validity metrics, and how to choose K (elbow, silhouette, gap statistic, stability).
- Where K-means fails: uneven sizes, densities and shapes. It is compared with hierarchical clustering and DBSCAN.
- Real applications: customer segmentation, colour quantisation and digit images.
- Explaining clusters, error analysis, and a deployable pipeline.

### [final_pycaret_zero_to_hero.ipynb](final_pycaret_zero_to_hero.ipynb)
PyCaret, the low-code ML library, explained from first principles using household budget data. It covers:
- The lifecycle and a first experiment.
- Every preprocessing option in `setup`.
- Comparing models, metrics and the plot gallery.
- Tuning, ensembling, stacking, calibration and cost-based thresholds.
- Regression and the leakage trap.
- Interpretability with SHAP.
- Clustering and anomaly detection.

### [final_pycaret_capabilities_tour_(1).ipynb](final_pycaret_capabilities_tour_(1).ipynb)
A fast tour of what PyCaret can do, with the same layout for each use case: the input, the one-line calls, the output, the picture, and when to use it. The use cases are:
- Binary and multiclass classification, and regression.
- Tune, blend and stack.
- Rare-event fraud detection.
- Clustering and anomaly detection.
- Time series.
- Text in tables.
- SHAP and fairness.
- Deployment.

### [final_autogluon_zero_to_hero.ipynb](final_autogluon_zero_to_hero.ipynb)
AutoGluon, the AutoML toolkit that trains and ensembles many models automatically, in 20 tasks. It covers:
- The three-line first predictor, reading the leaderboard and choosing metrics.
- Class imbalance, decision thresholds and calibration.
- Regression, quantile regression and the leakage trap.
- Presets, time budgets, bagging, stacking and hyperparameters.
- Feature importance and per-prediction explanations.
- Time series forecasting with Chronos.
- Multimodal text plus tabular data.
- MLOps: saving, refitting, distilling, speed and drift.

### [final_autogluon_capabilities_tour.ipynb](final_autogluon_capabilities_tour.ipynb)
A tour of AutoGluon across many problem types. Each uses the same few lines of code on a different dataset:
- Churn classification, loan grades and house prices.
- Quantile regression for delivery windows.
- Card fraud with cost-aware thresholds.
- Store demand forecasting.
- Text and tabular product reviews.
- Defect photos.
- Embeddings and semantic search over support tickets.

### [final_nvidia_rapids_zero_to_hero.ipynb](final_nvidia_rapids_zero_to_hero.ipynb)
NVIDIA RAPIDS, which runs the pandas and scikit-learn workflow on a GPU, in 15 or more tasks. Timed benchmarks show where the GPU is faster. It covers:
- Why GPUs help, and a first benchmark.
- cuDF as pandas on the GPU, and the zero-code-change `cudf.pandas` mode.
- Aggregation and plotting.
- cuML classification, regression and pipelines that stay on the GPU.
- Unsupervised learning on 200,000 households: k-means, DBSCAN, HDBSCAN, PCA, UMAP and t-SNE.
- Nearest neighbours and anomaly scores.
- Cross-validation and model search at speed with `cuml.accel`.

## Running the notebooks

The notebooks were built to run in Google Colab. The RAPIDS notebook needs a GPU runtime (a T4 is enough). The others run on a CPU, though AutoGluon is faster with a GPU. The notebooks are large because they are saved with their outputs, so they open already populated with results and plots.
