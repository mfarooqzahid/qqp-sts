# QQP STS Demo And Testing Data

Use this file as the demo checklist for presenting the project.

## Quick Start

```bash
python3 app.py
```

Open the local dashboard URL, usually:

```text
http://127.0.0.1:8501
```

Recommended demo model:

```text
TF-IDF + Logistic Regression
```

This model works offline and does not need to download Sentence-BERT weights.

## Demo Flow

1. Show the project title and dataset summary cards.
2. Explain that the app detects whether two Quora-style questions mean the same thing.
3. Run two or three examples from the testing table below.
4. Turn on model comparison and show how the TF-IDF baseline and Logistic Regression differ.
5. Show the evaluation metrics table, ROC curve, and confusion matrix selector.
6. Show the dataset summary and sample question pairs.
7. Explain that the Python server keeps the model pipeline separate from the HTML/CSS/JS interface.

## Testing Data

| ID | Question 1 | Question 2 | Expected Result | Why This Is Useful |
|---|---|---|---|---|
| T01 | How can I learn Python? | What is the best way to learn Python? | Duplicate | Clear paraphrase with same intent. |
| T02 | How do I improve my English speaking skills? | What is the best way to become fluent in spoken English? | Duplicate | Same topic with different wording. |
| T03 | What are the best places to visit in Japan? | Which cities should I see when traveling to Japan? | Duplicate | Travel intent is semantically similar. |
| T04 | How can I lose weight safely? | What are healthy ways to reduce body weight? | Duplicate | Health wording changes but meaning stays close. |
| T05 | What causes solar eclipses? | How do I make sourdough bread? | Not Duplicate | Completely unrelated topics. |
| T06 | Why is the sky blue? | What is the fastest way to learn JavaScript? | Not Duplicate | Strong negative example. |
| T07 | How do I prepare for a job interview? | What should I wear to a wedding? | Not Duplicate | Both are advice questions but different intent. |
| T08 | Can I use a credit card without a bank account? | How do banks approve home loans? | Not Duplicate | Same broad finance domain, different task. |
| T09 | How do I fix a slow laptop? | Why is my computer running so slowly? | Duplicate | Different phrasing, same troubleshooting intent. |
| T10 | What is machine learning? | How does artificial intelligence work? | Not Duplicate | Related domain but not exactly duplicate. |
| T11 | How can I start investing with little money? | What are beginner investment options for a small budget? | Duplicate | Good real-world paraphrase. |
| T12 | Is coffee bad for health? | What are the health effects of drinking coffee? | Duplicate | Same intent with neutral wording. |

## CLI Demo

Run a quick terminal prediction:

```bash
python3 cli.py "How can I learn Python?" "What is the best way to learn Python?"
```

Expected output shape:

```text
--- Result ---
Model: TF-IDF + Logistic Regression
Similarity Score: ...
Verdict: Duplicate
--------------
```

Try a negative example:

```bash
python3 cli.py "What causes solar eclipses?" "How do I make sourdough bread?"
```

Expected verdict:

```text
Not Duplicate
```

## Presentation Script

This project solves semantic textual similarity for Quora Question Pairs. Given
two questions, it predicts whether they are duplicates, meaning both questions
ask the same underlying thing.

The project compares three modeling strategies. The first is a simple TF-IDF
cosine similarity baseline. The second uses TF-IDF difference and product
features with Logistic Regression. The third uses Sentence-BERT embeddings with
cosine similarity.

For the live demo, I am using the Logistic Regression model because it works
offline and gives a stronger supervised baseline than plain cosine similarity.
Sentence-BERT is included as an optional model when the MiniLM weights are
available locally or network access is available.

The app is not only a prediction form. It also includes dataset statistics,
recent prediction history, side-by-side model comparison, evaluation metrics,
ROC curves, confusion matrices, and a short architecture view. This makes it a
complete project rather than only a model script.

## Questions To Prepare For

**Why use three models?**

To show progression from a simple baseline to a supervised model and then to a
modern embedding-based semantic model.

**Why does Sentence-BERT sometimes fail to load?**

It needs pretrained MiniLM weights. If they are not cached locally, the library
tries to download them from Hugging Face.

**Which model should be shown during the demo?**

Use TF-IDF + Logistic Regression because it is reliable offline.

**What metric matters most?**

ROC-AUC is useful for ranking quality, while F1 balances precision and recall
for the duplicate class.

**What is the main limitation?**

The dataset is a 12k sample, so a larger training set and threshold tuning would
improve production-level performance.

## Backup Plan

If the dashboard does not open, use the CLI:

```bash
python3 cli.py "How do I fix a slow laptop?" "Why is my computer running so slowly?"
```

Then show these files directly:

```text
pipeline.py
evaluate.py
outputs/metrics.csv
outputs/roc_curves.png
```
