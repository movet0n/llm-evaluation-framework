# LLM Evaluation Framework

A small but serious framework for testing whether a language model actually knows what it's doing.

The specific task here is toxicity classification — given a piece of text, does the model correctly identify it as toxic or safe?

## The core idea

You can't just ask a model if it's good at something and trust the answer. You need a dataset of examples where you already know the correct answer, run the model on all of them, and measure where it agrees with you and where it doesn't. That gap — between what the model says and what's actually true — is the whole point of evaluation.

This is what every serious AI lab does before shipping a model. We're doing a smaller version of that, locally, for free.

## What's actually being measured

The framework tracks four things per run:

**Accuracy** is the obvious one — what percentage did it get right overall. It feels like the most important number, but it's often the most misleading. If 90% of your dataset is safe text, a model that always says "safe" scores 90% accuracy while completely failing at its actual job.

**Precision** asks: when the model raises an alarm, how often is it right? Low precision means lots of false alarms — the model is trigger-happy.

**Recall** asks: of all the actually toxic texts, how many did the model catch? Low recall means it's missing real threats. For content moderation, this is the scary one.

**F1** is the balance between those two. It's the number to watch when precision and recall are in tension — which they almost always are.

## The dataset

Twenty examples, split into tiers by how hard they are:

- **clear_toxic** — "Go kill yourself." Nobody disputes this.
- **mostly_toxic** — "People like you are the reason nothing gets done." Clearly hostile, but less direct.
- **borderline** — "Keep pushing me and see what happens." Implicit threat. Plausible deniability. The interesting cases.
- **mostly_safe** — "I really disagree with your approach here." Critical, possibly uncomfortable, but not toxic.
- **clear_safe** — "Have a great weekend!" Completely fine.

The tiers matter because a model that only catches obvious toxicity isn't actually useful. The borderline tier is where models tend to fail, and where the real work is.

## A note on confidence scores

Each prediction includes a confidence score — the model rating its own certainty on a scale of 0 to 1. It sounds useful. It's more complicated than that.

The number isn't coming from inside the model's reasoning process. It's generated the same way any other word is — the model predicts what number makes sense to write there, based on context. We're essentially asking it to self-assess, and then trusting its answer.

For strong models at low temperature, this correlates reasonably with correctness. For local models, especially when temperature is turned up, it's mostly noise — you'll see the label stay completely stable across runs while the confidence number swings from 0.15 to 0.95 and back. The label is the real signal. The confidence score is the model's best guess at what a confident-sounding number looks like.

The real confidence a model has lives in something called logprobs — the actual probability it assigned to each token at generation time. That's a true internal measurement. But it requires lower-level API access and is harder to work with, so most evaluation pipelines, including this one, use prompted self-assessment instead and take the numbers with appropriate skepticism.

## Stability analysis

LLMs are probabilistic. Run the same text through the same model twice and you might get different answers — especially on ambiguous cases. The stability analysis runs each sample multiple times and measures how often the model changes its mind.

A model that's 100% consistent is either very good or very stubborn. A model that flips on borderline examples is telling you something honest: it doesn't actually know. That's useful information.

You can control how many runs to do and what temperature to use:

```bash
# Single eval, deterministic
python -m runners.eval_llm_classifier

# Stability analysis — 5 runs per sample at temperature 0.7
EVAL_RUNS=5 python -m runners.eval_llm_classifier
```

Reports get saved automatically to `reports/` as timestamped JSON files, one per run.

## Running it

You'll need [Ollama](https://ollama.com) installed and running, with a model pulled:

```bash
ollama pull llama3.2
ollama serve
```

Then from the project root:

```bash
pipenv install
python -m runners.eval_llm_classifier
```

The framework also works with OpenAI — set `OPENAI_API_KEY` in your `.env` and remove `OLLAMA_BASE_URL`. The code doesn't care which end it's talking to.

## Project structure

```
datasets/        — the labeled test examples
evaluators/      — classification logic, metrics, stability analysis
runners/         — the main script that ties it all together
reports/         — output from each run (gitignored)
```

## What this is good for

Testing one model on one task and understanding exactly where and how it fails. Not a benchmark suite, not a leaderboard — just a clear, honest measurement of whether the model does what you want it to do.

That turns out to be surprisingly rare.

## Architecture Diagram

┌─────────────────────────────────────────────────────────────────────┐
│  python -m runners.eval_llm_classifier                              │
│                         main()                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ reads env vars
                               │ OLLAMA_MODEL, EVAL_RUNS,
                               │ EVAL_AGREEMENT_THRESHOLD,
                               │ EVAL_STABILITY_TEMPERATURE
                               │
          ┌────────────────────┼───────────────────────────┐
          │                    │                           │
          ▼                    ▼                           ▼
┌──────────────────┐  ┌─────────────────────┐   ┌──────────────────────────┐
│ datasets/        │  │ evaluators/         │   │ evaluators/              │
│ toxicity_v1.py   │  │ classifier.py       │   │ stability.py             │
│                  │  │                     │   │                          │
│  DATASET[]       │  │ classify_with_llm() │   │ run_stability_analysis() │
│  LABELS[]        │  │                     │   │                          │
└────────┬─────────┘  └───────┬─────────────┘   └───────────┬──────────────┘
         │                    │                             │
         │  20 labeled rows   │  → (label, conf)            │
         │                    │                             │
         └──────────┬─────────┘                             │
                    │                                       │
                    ▼                                       │
     ┌──────────────────────────┐                           │
     │  IF n_runs == 1          │                           │
     │  (single eval mode)      │                           │
     │                          │                           │
     │  for row in DATASET:     │                           │
     │    classify_with_llm()   │                           │
     │    → predictions[]       │                           │
     └──────────┬───────────────┘                           │
                │                                           │
                │         IF n_runs > 1                     │
                │         (stability mode)                  │
                │                                           │
                │    DATASET ───────────────────────────────┘
                │                    │
                │                    │  classify_fn = lambda text:
                │                    │    classify_with_llm(client, text,
                │                    │      model, temperature=0.7)
                │                    │
                │                    ▼
                │         run_stability_analysis(
                │           DATASET, classify_fn, n_runs=5
                │         )
                │           │
                │           │  for each sample:
                │           │    runs classify_fn × n_runs
                │           │    → [SampleStability]
                │           │       .majority_label
                │           │       .agreement_rate
                │           │       .mean_confidence
                │           │       .std_confidence
                │           │       .unstable
                │           │
                │           └──→ predictions[] (derived from majority_label)
                │
                ▼
     ┌───────────────────────────────────────────────┐
     │  predictions[]  (from either path above)      │
     │  [{ id, text, gold, tier, pred, conf }, ...]  │
     └──────────────┬────────────────────────────────┘
                    │
          ┌─────────┼──────────────┐
          │         │              │
          ▼         ▼              ▼
   outcome()   confusion_      print_tier_
   per row     from_           breakdown()
   → TP/FP/    predictions()   │
     TN/FN     │               │  for each tier:
               │               │    confusion_from_
               ▼               │    predictions()
          ┌────────────┐       │    → Metrics
          │ Metrics    │       │
          │ .tp .fp    │       └──→ tier_results{}
          │ .tn .fn    │
          │            │
          │ .accuracy  │
          │ .precision │
          │ .recall    │
          │ .f1        │
          └────┬───────┘
               │
               ▼
        save_report()
        ┌─────────────────────────────────┐
        │ reports/                        │
        │ 20250222T143021Z_llama3.2.json  │
        │ {                               │
        │   predictions[],                │
        │   confusion_matrix{},           │
        │   metrics{},                    │
        │   tier_metrics{},               │
        │   stability{}  (if n_runs > 1)  │
        │ }                               │
        └─────────────────────────────────┘
