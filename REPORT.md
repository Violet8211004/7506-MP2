# Selecting One of Four Candidate Solutions

**DASE7506 Mini-Project 2**  
[Yushuo Zhang], [3036708270]  
31 October 2026

The number I am submitting is the official scorer’s `accuracy` field: **80.51391862955032** (376/467).

The program does not write a new solution. For each question it scores the four given candidates and returns one index in `{0, 1, 2, 3}`. The submitted score is a mix of three models. A LightGBM ranker and a pairwise linear model are standardised inside each question and combined with ranker weight 0.45. A small listwise network is then mixed in at weight 0.25. I chose those weights on the validation set after the training settings were frozen. The test set was scored only after each kept method was frozen. Earlier test scores were 77.08779443254818, 78.80085653104925 and 79.01498929336188. I did not use those numbers to change a model.

## 1. Who did what

I wrote the improvement plan in `DASE7506_MP2_正确率优化策略.md`. The idea was to stop treating the four candidates separately: use the vote among their final numbers, check simple arithmetic, rank them with the supplied embeddings, and combine models only when they make different mistakes. I also decided that the useful validation number is not the headline accuracy. The test set has at least two different parsed answers in every question, so I asked for the same slice of the validation set to be reported throughout.

A coding agent in Cursor implemented that plan, ran the experiments, and drafted this report from the result files. I set which comparisons to run and when a gain was large enough to keep. I checked the official score files. I am responsible for the submitted program and for the claims below.

## 2. Task and data

The bank protocol is `dase7506-mp2-open-hard-verifier-v5`. There are 4,755 training questions, 798 validation questions and 467 test questions. They do not overlap. Each question has four candidate solutions from Qwen3-0.6B and a 1024-dimensional vector for each candidate. The scorer, `common.parse_answer`, reads the number after the last `####`, or the last `\boxed{...}` if there is no `####`. Fractions and decimals are compared as rational numbers. Any candidate with the right final number scores, even if an earlier sentence is wrong. A candidate that does not parse cannot be correct.

On the validation set, 379 of 798 questions have all four candidates correct, so every method gets them. The test set has no such questions. I therefore also count two harder slices. “Hard” means fewer than four correct candidates (419 validation questions). “Disagreement” means at least two different parsed numbers (355 validation questions). The disagreement slice is the one I treat as comparable to the test set.

Of 14,353 positive training labels, 10,067 come from questions with only one distinct parsed number. A model trained on overall accuracy spends most of its positive gradient on questions that will not appear in the test set. That is why several of my later choices train or stop on the disagreement slice instead.

Validation labels were used to compare methods I had already specified, and to choose blend weights. The test references were opened four times, once for each frozen method in Section 6. No test score was used to pick the next recipe. The audit split was not used for selection. I did check that the submitted function returns one Python `int` on banks of 1, 5 and 17 questions, so it does not assume a fixed number of questions.

## 3. What I expected

I expected three things.

First, the supplied embedding should beat a vote on the final number, and both should beat surface cues such as length and confident wording. The embedding was built from the whole trace. Length was not.

Second, a score that looks at one candidate at a time should be weaker on questions where the four answers disagree. Comparing a correct candidate with an incorrect one, and training mostly on those questions, should move the disagreement slice more than it moves the easy questions.

Third, a tree model that also sees the vote and a few arithmetic checks should miss different questions from the linear embedding score. If that happened, mixing the two scores would be worth the extra code. A small neural ranker would be worth adding only if it was complementary on validation, not merely because it is a neural net.

The comparisons below are the tests of those expectations. Failed rows stay in the tables.

## 4. First comparison

The development slice used below is a seed-0 shuffle of the training questions: 475 held out, 4,280 in the body. E1 to E6 are fit on all 4,755 training questions, so that E1 matches a fixed recipe. E7 to E9 may use the holdout. None of E2 to E9 was allowed to replace E1 unless the pre-written rule said so. For E7 to E9 that rule was a gain of at least 0.5 points overall on the holdout, or 2 points on questions whose vote lead is at most one, without falling below E1 on validation. For E2 to E6 the rule was the same kind of gain on validation itself.

| ID | Idea |
|---|---|
| E0 | Majority vote on the parsed number. Ties take the smallest index. |
| E1 | Pointwise logistic regression on the 1024-dimensional embedding. 80 steps, learning rate 0.2, L2 0.01 on the weights, zero initialisation. Unparsed candidates are removed at prediction. |
| E2 | The same linear model on length, bold markers, boxes, check marks and confident phrases. |
| E3 | The same linear model on the vote count. |
| E4 | Hashed full text, dimension 2^16, hash seed 0. |
| E5 | The same hash, but only the final-answer span. |
| E6 | Arithmetic-check rate, coverage of question numbers, and the vote. |
| E7 | Pairwise logistic regression on embedding differences, stopped on overall development accuracy. |
| E8 | A grid mix of the E1 score and the vote, coefficients chosen on the development slice. |
| E9 | One hidden layer of width 64, seed 0. |

| Method | All | Hard | Tie (88) | Lead 1 (119) |
|---|---:|---:|---:|---:|
| E0 vote | 86.22% (688/798) | 73.75% (309/419) | 38/88 | 81/119 |
| E1 embedding | 90.10% (719/798) | 81.15% (340/419) | 58/88 | 89/119 |
| E2 surface | 78.95% (630/798) | 59.90% (251/419) | 30/88 | 56/119 |
| E3 votes only | 86.22% (688/798) | 73.75% (309/419) | 38/88 | 81/119 |
| E4 full-text hash | 83.33% (665/798) | 68.26% (286/419) | 37/88 | 69/119 |
| E5 answer-span hash | 76.19% (608/798) | 54.65% (229/419) | 32/88 | 51/119 |
| E6 checks + votes | 86.84% (693/798) | 74.94% (314/419) | 43/88 | 81/119 |
| E7 pairwise | 89.97% (718/798) | 80.91% (339/419) | 54/88 | 89/119 |
| E8 score + votes | 90.10% (719/798) | 81.15% (340/419) | 58/88 | 89/119 |
| E9 width-64 net | 86.72% (692/798) | 74.70% (313/419) | 53/88 | 77/119 |

No row replaced E1. E7 was the closest. It scored 433/475 on the development slice against 424/475 for E1, but validation was 718/798, one question worse, so I rejected it. E8’s best vote weight was zero, so it copied E1. E3 copies the vote exactly. Surface features and hashes are worse than the embedding, which is what I expected if the useful signal is already in the 1024 coordinates.

E1’s float32 checkpoint is 12,292 bytes and trained in under a second on my CPU. E9 stores about 271 KB and scores 86.72%. Extra width did not pay. The official scorer confirms E0 at 688/798 and E1 at 719/798.

After this table was frozen I ran E1 once on the test set: 360/467, accuracy 77.08779443254818. On the 355 validation disagreement questions E1 scores 276/355, or 77.75%. The test result sits next to that slice, not next to 90.10%. The gap is mostly the removal of the easy groups.

## 5. Training on questions that look like the test set

E7 had trained on every question and stopped on overall development accuracy. I changed the question, not the model size. The new rule uses the same 206 development questions that have at least two parsed numbers. A recipe has to beat body-trained E1 by at least 2 points on that slice. Body-trained E1 scores 154/206 at step 80.

The grid keeps the learning rate at 0.2 and at most 160 steps. It varies the training rows, whether embeddings are centered inside the question, the L2 coefficient (0.01, 0.1, 1), and pointwise versus pairwise loss. Ties keep the earlier step, then the larger L2. The winner is pairwise loss, mixed questions only, L2 0.1, step 12: 170/206. Centering is worse (at best 147/206). L2 0.01 reaches the same 170/206; the tie-break keeps 0.1.

I then refit those settings on every mixed training question.

|  | Disagreement | Hard | All |
|---|---:|---:|---:|
| E1 | 276/355 | 340/419 | 719/798 |
| Pairwise, mixed questions, 12 steps | 285/355 | 349/419 | 728/798 |

The official validation accuracy is 91.2280701754386. All nine extra questions are in the disagreement slice. Ties stay at 58/88. The one-vote lead moves from 89/119 to 91/119. One test run then gave:

```text
correct: 368
n: 467
accuracy: 78.80085653104925
```

That is eight more than E1. I did not use this number to choose the next model.

## 6. Ranker, arithmetic checks, and a neural ranker

The next part follows the plan: put the vote, simple arithmetic checks and the embedding into one ranker, then add another model only if the errors differ.

LightGBM lambdarank treats each question as a group of four. Features are the vote, text counts, counts of checked and wrong equations, cosine geometry of the four vectors, the pairwise logit, and 64 principal components fit on the training split only. Early stopping uses disagreement accuracy on the same 206 development questions. The kept round is refit on all 4,755 training questions. The main settings are 31 leaves, learning rate 0.05, depth 5, minimum child samples 30, feature fraction 0.8, seed 42.

| Ranker | Disagreement | All |
|---|---:|---:|
| Vote and text only | 253/355 | 696/798 |
| Embeddings, no arithmetic checks | 283/355 | 726/798 |
| Full features | 286/355 | 729/798 |
| Full features, mixed questions up-weighted ×4 | 286/355 | 729/798 |
| Full features, binary objective | 268/355 | 711/798 |
| Pairwise linear score | 285/355 | 728/798 |

Arithmetic checks add 3 disagreement questions over the embedding-only ranker. Up-weighting every mixed question does nothing further. Binary classification is worse. The full ranker is only one question ahead of the pairwise score, so I did not replace the linear model with it. On the disagreement slice, 24 questions are right only for the ranker and 23 only for the linear score. That was the reason to mix them.

Each question’s four scores are standardised, then combined. Ranker weights 0.35, 0.45 and 0.50 tie. I kept the middle value, 0.45. Validation becomes 289/355 and 732/798. The official accuracy is 91.72932330827068. One test run gave 369/467, accuracy 79.01498929336188, one more question than the pairwise model.

The listwise network is the other model in the plan. It is a shared layer of width 128, a ReLU, dropout 0.1 in training, and a comparison with the group mean. The loss puts probability on the set of correct candidates, because more than one candidate may be right. Each epoch permutes the four positions together with their labels. Training uses mixed questions only. Epoch 6 is best on the development slice (164/206). Alone, validation is 285/355 and 728/798, which does not beat the blend.

The errors still differ: 22 disagreement questions are right only for the network, 26 only for the blend. Weights 0.25 and 0.30 both reach 298/355 and 741/798, and every weight from 0.15 to 0.70 is also above the blend. I kept the smaller network weight, 0.25. The official validation accuracy is 92.85714285714286. Hard questions move from 353/419 to 362/419. One test run gave:

```text
correct: 376
n: 467
accuracy: 80.51391862955032
```

That is seven more than the ranker–linear blend, and 16 more than E1. This fourth test score was not used to change the weight.

## 7. Changes that did not help

I had written down several other settings before seeing these results. They were run the same way: choose the round on the development disagreement slice, refit, then score validation once. None of them replaced the submitted mix.

| Variant | Disagreement | All |
|---|---:|---:|
| PCA width 128 | 280/355 | 723/798 |
| 15 leaves | 282/355 | 725/798 |
| 15 leaves, learning rate 0.03, depth 4 | 281/355 | 724/798 |
| Time, percent and money checks added | 279/355 | 722/798 |
| 32 components of the group-centered embedding | 271/355 | 714/798 |

Explicit unit conversions such as “2 hours 20 minutes = 140” appear in only 3.45% of training candidates, so the ranker has little to split on. The Jay concert example in my plan, where 1 hour 20 minutes is misread as 90 minutes, is rare in this form. Mixing the 15-leaf ranker with the pairwise score reaches 292/355 and 735/798, one question short of the bar I had set for another test run.

After the three-model mix was frozen, I still checked four ideas on validation and did not open the test set again.

- Reweighting the three scores directly, instead of nesting the mixes, reaches 301/355 and 744/798 at one weight only. Neighbouring weights are lower. I did not keep it.
- A ranker-plus-network mix without the linear score reaches 294/355 and 737/798.
- Averaging softmaxes at a shared temperature reaches 300/355 and 743/798, again without a stable plateau above the bar.
- A student network trained on the blend’s soft training scores reaches 290/355 and 733/798. The blend already runs in about a second, so distillation was not needed for the time limit either.
- A linear model trained only on questions whose vote lead is at most one reaches 284/355 and 727/798. Every positive mix weight is worse than the blend alone.
- Up-weighting the 179 training questions the blend still misses, by 4× or 8×, gives a ranker at 289/355 and 732/798. It fixes 14 disagreement questions the blend misses and loses 23 that the blend gets. Mixing them does not help.

Temperature scaling of a single model cannot change its own top-1 choice, so I never ran it as a method by itself.

## 8. Where it still fails

These examples are from the validation set. I did not tune on them.

On a hiking question the right load is 40 pounds, because 100 miles need 50 pounds of supplies and the first pack is `50 / 1.25`. The submitted style of embedding score prefers a long candidate that computes 100 and 50 correctly and then multiplies by `(1 - 0.25)`. The parser reads `75/2`. The trace looks orderly. Length is not evidence.

On Jay’s concert the right number of songs is 13. One selected candidate starts from the wrong clock times and then solves `5(x - 1) + 10 = 70`, which does give 13. The scorer awards the point. The method is selecting a final number, not checking every sentence.

John’s naps are a 2–2 tie between 60 and 420. The vote takes the earliest index and gets 60. The pointwise embedding score takes a trace that treats 70 days as 70 weeks and gets 420. The vote is right on 17 disagreement questions where the final blend is wrong, but the blend is right on 70 where the vote is wrong. Switching back to the vote on ties would give up more than it saves: on tied questions the blend alone is right on 35 and the vote alone on 12.

Of the 57 disagreement questions the final blend still misses, 46 are ties or one-vote leads, and 40 are also missed by the vote. Only 4 have a correct candidate with a cleaner equation check than the chosen one. The remaining mistakes are mostly a wrong reading of the question, not a failed addition. That is why more arithmetic patterns and another small linear model did not move the validation numbers.

## 9. Cost

The submitted checkpoint is `code/runs/model_bundle_blend_mlp/`. It is 1,278,868 bytes: `mlp.npz` 542,846, `model.txt` 441,825, `preprocess.npz` 280,344, `linear.npz` 13,521, `config.json` 332. The limit is 16 MiB. Earlier checkpoints stay under `code/runs/` so the three earlier test scores can be reproduced. The `runs/` directory is gitignored, so the checkpoint has to be submitted with the code rather than assumed to be in the repository.

I measured the submitted test run with four threads on an AMD Ryzen 7 6800H, Python 3.12.10, NumPy 2.5.3 and LightGBM 4.7.0. Wall time was 1.22 seconds for prediction and 0.28 seconds for scoring. The JSON field `prediction_seconds` is 0.900. The reference limit is 30 seconds on a Xeon Platinum 8457C. This machine is not that CPU. No GPU was used.

The libraries are pinned in `code/requirements.txt`: NumPy 2.5.3, LightGBM 4.7.0, SciPy 1.18.1. SciPy is installed with LightGBM. I did not change `common.py`, `predict.py`, `score_predictions.py` or `check_data.py`.

From `code/`, with the virtual environment:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
python tune_generalize.py
python train_lgbm.py
python train_unused.py
python predict.py --method submission.py --checkpoint runs/model_bundle_blend_mlp --split test --output runs/test-blend-mlp-predictions.json
python score_predictions.py --split test --predictions runs/test-blend-mlp-predictions.json --output runs/test-blend-mlp-score.json
```

`tune_generalize.py` writes the pairwise checkpoint and reads validation only after the development rule has chosen the recipe. `train_lgbm.py` writes the five rankers and scores validation only after they are refit. `train_unused.py` writes the variants in Section 7. The submitted directory copies the ranker–linear blend plus `listwise_mlp/mlp.npz`, with `config.json` set to method `blend_mlp`, `mlp_alpha` 0.25 and `rank_alpha` 0.45. `train_linear.py` reproduces E1. `run_experiments.py` reproduces the E0–E9 table. The last two commands are the test run and should be used once.

SHA-256 of the submitted checkpoint:

- `model.txt`: `904CF10D3A1B089352CA9039E342AB90EF369CE943D0878FFA69C4E04654637C`
- `preprocess.npz`: `7C252C3E903E53F9661F0FB463A845E1D73D8BF02ECA44D66A6CAC3234A31CF4`
- `linear.npz`: `58E831C0CE8EBE58B5981C0BD4E3D2D25DD8ADD9F547D39116BD5BD289E0EE6F`
- `mlp.npz`: `4C8989DCCA421AB4F96AA835196AD463F79F2D034B0734C4B1528F10D6A22CC0`
- `config.json`: `97176465C8DAAD7B9C70409DE24374B5605272EEE3CC3860AE4366D2CAA1354E`

## 10. Conclusion

Reading the trace through the supplied embedding helps: the vote scores 86.22% on validation and the pointwise embedding score 90.10%. Surface wording does not explain that gap. Judging one trace at a time is not enough once the test set drops the easy questions. E1’s test accuracy, 77.09%, matches its 77.75% on validation questions whose answers disagree, not its 90.10% headline.

Training a pairwise score only on those disagreeing questions raises the test score to 78.80%. A ranker that also sees the vote and arithmetic checks is complementary, but by itself it is only one validation question better. Mixing it at weight 0.45 reaches 79.01% on the test set. A listwise network does not win alone, but it corrects 22 validation questions the blend misses. Mixing it at weight 0.25 is the submitted method, at 80.51%. The extra compute is small: about a second and a half, and about 1.2 MB.

Further changes from the same plan, including wider PCA, unit conversions, distillation, and models aimed only at close votes or at the blend’s remaining training errors, did not produce a stable validation gain. I left the checkpoint unchanged after those checks.
