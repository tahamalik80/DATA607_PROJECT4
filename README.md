# Project 4 — Document Classification (Spam vs Ham)

Author: Taha Malik  

---

## Overview

This repository contains an R Markdown analysis that implements a full end-to-end document classification pipeline to identify spam vs ham (non-spam) email messages. The pipeline uses the SpamAssassin public corpus (or any similarly structured spam/ham folders) and demonstrates:

- data ingestion from directory of raw email files
- preprocessing and tokenization using quanteda
- exploratory data analysis (message length, top terms, word clouds)
- feature engineering (term-frequency and TF‑IDF)
- model training and evaluation (Naive Bayes baseline and Random Forest)
- evaluation metrics (confusion matrices, precision/recall/F1, ROC/AUC)
- error inspection and interpretation
- guidance for improvements and reproducibility

The main executable file is:
- `Document_Classification_Project4.Rmd` — knit to HTML/PDF to view the complete report, code and figures.

This README explains how to run the analysis, key design choices, outputs to expect, and suggestions for extensions.

---

## Requirements

R (>= 4.0 recommended) and the packages below. You can install packages with `install.packages()`.

Required R packages:
- tidyverse
- quanteda
- quanteda.textplots
- caret
- e1071
- naivebayes
- randomForest
- pROC
- wordcloud
- knitr
- kableExtra
- ggplot2
- ggrepel
- RColorBrewer

Install example:
```r
install.packages(c(
  "tidyverse","quanteda","quanteda.textplots","caret","e1071",
  "naivebayes","randomForest","pROC","wordcloud","knitr",
  "kableExtra","ggplot2","ggrepel","RColorBrewer"
))
```

Note: `quanteda` relies on compiled code and may require a working build environment on some systems. Use your OS package manager / prerequisites if installation fails.

---

## Data

This project expects two directories containing raw email files (one file per email):

- spam folder (example): `C:/Users/taham/OneDrive/Documents/Data 607/Project 4/20050311_spam_2/spam_2`
- ham folder (example): `C:/Users/taham/OneDrive/Documents/Data 607/Project 4/20030228_easy_ham/easy_ham`

These folders come from the SpamAssassin public corpus (https://spamassassin.apache.org/old/publiccorpus/). The original format has compressed `*.tar.bz2` files that need to be extracted; some systems require double extraction as described in the instructor video transcript.

Important:
- Each email should be an individual file in the folder.
- Remove any extraneous files like `cmds` (the Rmd includes a filter to exclude files named `cmds`).

If you need to re-create the sample dataset, download the `easy_ham` and `spam` archives, extract them, and place the resulting email files into two directories.

---

## How to run

1. Edit the top of `Document_Classification_Project4.Rmd` to set the paths:
```yaml
params:
  spam_dir: "C:/path/to/your/spam_folder"
  ham_dir:  "C:/path/to/your/ham_folder"
```

2. Open the R Markdown file in RStudio and click Knit, or run programmatically:
```r
rmarkdown::render("Document_Classification_Project4.Rmd", output_format = "html_document")
```

3. The Rmd will:
- read the files,
- build a quanteda corpus and dfm,
- run EDA and produce plots and wordclouds,
- train Naive Bayes (TF) and Naive Bayes (TF‑IDF) models,
- train Random Forest on top‑K features,
- show confusion matrices, precision/recall/F1 and ROC/AUC,
- print model comparisons and save models if you enable the save block.

---

## Pipeline summary (what the Rmd does)

1. Read raw email files, collapse multi-line email files into a single text value per document, and attach a `label` (spam/ham).
2. Basic EDA:
   - Calculate number of words and characters per message.
   - Plot histograms (log-scaled).
   - Provide descriptive tables of length statistics.
3. Preprocessing / DFM creation:
   - Lowercasing, remove punctuation, numbers and separators.
   - Remove English stopwords.
   - Stem tokens (word stemming).
   - Create a term-frequency dfm.
   - Trim features by document frequency (min 0.5% of docs; max 99%).
4. Exploratory features:
   - Top terms overall and per-class.
   - Wordclouds overall and per-class.
5. Feature engineering:
   - Term-frequency (TF) matrix (baseline).
   - TF‑IDF matrix (baseline comparison).
6. Modeling:
   - Naive Bayes on TF (baseline).
   - Naive Bayes on TF‑IDF (baseline).
   - Random Forest trained on top‑K frequent features (K adjustable; default 1000).
7. Evaluation:
   - Confusion matrices, precision, recall, F1.
   - ROC curves and AUC.
   - Error inspection (false positives and false negatives).
8. Interpretation and conclusions:
   - Narrative analysis describing the models’ performance, common error patterns and feature importances.

---

## Design choices and rationale

- Stemming: reduces vocabulary size and sparsity by conflating inflected forms.
- Stopword removal: removes high-frequency function words that rarely contribute discriminative power.
- Document frequency trimming: excludes extremely rare tokens (noise) and extremely common tokens (uninformative).
- TF baseline + TF‑IDF baseline: TF is a simple and fast baseline; TF‑IDF helps emphasize discriminative terms and reduces weight of common terms.
- Naive Bayes: a fast probabilistic baseline commonly used in text tasks.
- Random Forest: a non-linear ensemble method that can capture interactions between terms and tends to perform well on sparse high-dimensional features once dimensionality is controlled.

---

## Key files and outputs

- Document_Classification_Project4.Rmd — the reproducible analysis notebook (main entry point)
- (optional) saved model files if you enable and run the save block:
  - naive_bayes_model_tf.rds
  - naive_bayes_model_tfidf.rds
  - random_forest_model.rds
- Figures and tables produced when knitting:
  - histogram of message lengths
  - top-term bar plot
  - wordclouds (overall and per-class)
  - confusion matrices (NB, NB-TFIDF, RF)
  - ROC curves (NB, NB-TFIDF, RF)
  - RF feature importance bar chart
- Session info is printed at the end for reproducibility.

---

## Expected results and interpretation

When run on the same SpamAssassin folders used in the Rmd, typical results seen in the notebook are:

- Naive Bayes (TF) — strong baseline, sometimes with high recall for spam but lower precision (more false positives). Moderate AUC (e.g., ~0.85).
- Naive Bayes (TF‑IDF) — TF‑IDF can shift precision/recall tradeoff; results depend on trimming and tokenization.
- Random Forest — strong performance when trained on a top‑K subset of features (e.g., K=1000). High precision and recall, AUC close to 1.0 on the test split, and easily interpretable top features (e.g., "click", "free", "href", "html", server tokens). This model tends to outperform Naive Bayes in this dataset.

Note: numeric metrics (accuracy, precision, recall, F1, AUC) will vary with dataset splits, trimming parameters, and top‑K choice for Random Forest. The Rmd includes a reproducible seed for splits (set.seed(123)).

---

## Troubleshooting & performance tips

- Memory usage: converting the trimmed dfm to a dense matrix with `as.matrix()` can be memory-intensive if there are many features. If you run out of memory:
  - Increase `min_docfreq` to remove more rare terms.
  - Reduce the number of features used by Random Forest (lower K).
  - Use sparse-aware models (e.g., glmnet or text-specific frameworks) or keep quanteda dfm and use models that accept sparse matrices.
- Package install errors: make sure system dependencies (Rtools on Windows, build-essential on Linux, Xcode on macOS) are installed for package compilation.
- Encoding problems: the Rmd includes attempts to read with both UTF-8 and latin1; if you still see errors, ensure file encoding in your OS and R session are correctly set.
- Slow knit time: the Random Forest step can be the slowest. Reduce `ntree` or limit features during development.

---

## Recommendations for extension & deployment

- Add n-grams: include bigrams/trigrams or character n-grams to capture phrases and obfuscated text.
- Hyperparameter tuning: use `caret` or `tidymodels` to tune RF (mtry, ntree) or try XGBoost / LightGBM for better speed/accuracy.
- Class weights & calibration: if operating under strict cost for false positives/negatives, apply class weights or threshold tuning.
- Production pipeline: serialize the final model and create a small pre-processing wrapper to accept raw email text and return a score or label. Add unit tests and CI pipelines for model packaging.
- Use an out-of-time holdout or cross-validation for robust performance estimates.

---

## Reproducibility

- Seed: `set.seed(123)` is used before the train/test split.
- Session info: the Rmd prints `sessionInfo()` at the end; keep this block when sharing results.
- Documented parameters: change only the `params$spam_dir` and `params$ham_dir` in the Rmd header for portability.
