# Emotion Tracking Filters

This repository contains reproducible experiments for comparing different statistical filtering models in the context of short-term emotion score tracking.

## Overview

This project evaluates and compares three methods for tracking emotion scores over time:

- Kalman Filter
- Bayesian Mean Update
- Moving Average Filter

All models are tested on 7-day emotional sequences that simulate mood recovery, starting from low (depressive) values and stabilizing into normal range.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

```bash
python experiment.py
```

## Output

Generates a graph under `figures/emotion-tracking-example.png` and prints:

- Mean Squared Error (MSE)
- Pearson Correlation Coefficient (r)

## Citation

This work is part of a poster project on statistical modeling of emotion scores.

## Acknowledgment

This research was supported by the MSIT(Ministry of Science and ICT), Korea, under the National Program for Excellence & Communication in SW(2021-0-01409) supervised by the IITP(Institute for Information & Communications Technology Planning & Evaluation).
