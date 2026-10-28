# Phase 1 evaluation contract

Ratings >= 4 are relevant positives. All ratings, including dislikes, count as seen history. No click, watch-time or conversion observations are inferred. The model ranks every movie observed in training by positive rating count, breaking ties by movie ID; zero-positive training movies remain eligible. Movie metadata alone does not establish historical availability.

Global 80/10/10 time quantiles define cutoffs. Events at each cutoff go to the later period, so timestamp ties never cross boundaries. Fractions can differ from nominal ratios. This avoids cross-user future interactions leaking into training, at the cost of many cold-start users. Every split must be nonempty.

Both validation and test use the **same frozen training model**. Validation excludes training-seen movies. Test excludes training and validation-seen movies, representing known history at the start of the test window. Neither target ratings nor target timestamps are used for ranking. No within-window feedback update, retraining, or test-driven tuning occurs.

Full training-catalog candidates, no sampled negatives. Relevant held-out movies absent from the training catalog remain in recall and NDCG denominators, making cold-item misses explicit. Users with no positive target ratings do not have defined recall and are reported separately. Users with no training history receive the same popularity fallback. Macro averages weight eligible users equally. Precision divides by requested K even when fewer candidates remain; ideal DCG uses min(K, number of relevant items). Hit rate is any relevant recommendation.

Catalog coverage is unique recommended / training catalog size. Popularity bias is mean training-positive count of recommended items (a descriptive exposure measure, not causal bias). Diversity is mean pairwise genre Jaccard distance; missing genre sets use distance 1. Novelty is mean -log2 of Laplace-smoothed training-positive item frequency. These proxies are not direct measures of satisfaction. The 95% percentile intervals resample users with replacement and a fixed seed; they describe user sampling variation conditional on this split/model, not uncertainty across time or trainings. Coverage is a corpus metric with no bootstrap interval in Phase 1.

Latency and throughput measure only sequential in-process recommendation calls, excluding evaluation bookkeeping, serialization, transport and concurrency. They are machine-dependent microbenchmarks, not API SLO evidence.

EDA covers the complete observed dataset for description only; its outputs never enter model fitting. Full 32M loading is supported but in-memory and not yet scale-validated; copies, grouping and CSV artifacts need substantial RAM/disk. 32M demographics are absent, never synthesized. Dataset validation fails on duplicates rather than silently choosing which observation to keep.
