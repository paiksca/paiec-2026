# PAIEC 2026: a benchmark-level prior learned from 182 public benchmarks

Code for an entry to the [NeurIPS 2026 Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition) (Codabench 17828) by Samuel Paik-Heintz, Institute for Systems Biology.

The predictor `predict(input, labeled)` returns the probability that an AI system (the subject) answers a benchmark item correctly. It sees 0 to 31 revealed labels, on a benchmark absent from the public data. It is a hierarchical logistic model whose posterior is refitted from `labeled` on every call:

    eta = mu_b + theta_s + delta_sb + g + u_group + e_i

- `mu_b` is the benchmark level. Its prior mean comes from a ridge regression of fitted benchmark levels on features of the item text and the subject's harness, trained over the public pool.
- `theta_s` is the subject standing, keyed on the canonical model name and learned across benchmarks. Unseen names fall back to an attribute regression on release date, provider, harness and reasoning effort.
- `delta_sb` is the pair's own deviation.
- `g` is a level shift shared by every benchmark in a run.
- `u_group` holds item_features group effects.
- `e_i` is the item residual, integrated out.

The submitted archive needs numpy only.

## Results

Grouped 8-fold cross-validation over benchmarks, on simulated runs shaped like 55 public formative feedback logs. Evaluation pairs are subjects released in 2025 or later, ALC is averaged per benchmark, and differences are paired with SEs clustered by benchmark.

| model | main regime (64 benchmarks) | hard regime (19 benchmarks under 40% accuracy) |
|---|---|---|
| smoothed label mean | 0.2125 | |
| hierarchical, constant level prior | 0.2106 | |
| hierarchical, feature-based level prior | 0.2078 | 0.2029 |
| + run shift + item_features groups (v2) | 0.2076 | 0.1985 |
| + level centre lowered by 1.0 (v3, submitted) | 0.2108 | 0.1887 |

v3 was chosen by a rule written down before the comparison: minimize the mean of the two regimes. The feature-based level prior hurts when fitted on 5 to 20 training benchmarks and helps from about 80 (`results/learning_curve.csv`). Repeated platform draws are in `results/submissions.tsv`.

## Reproduction

Python 3.11, laptop CPU. Every step runs on two processes.

1. Install: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
2. Data: accept the gated terms of [aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db) with your own Hugging Face account. Export `HF_TOKEN`, then run `.venv/bin/python scripts/fetch_data.py` and `.venv/bin/python scripts/consolidate.py`. This pins revision `54074453da7b9911be0f48e4c560be547bc84d21` and writes `data/consolidated/` (about 200 MB). The data are not redistributed here. `data/hf/` can be deleted afterwards.
3. Cross-validation: `.venv/bin/python scripts/local_run.py` runs every command in `scripts/jobs.txt` (30 CV runs, a few hours on two cores) and skips outputs that exist. Summarize one result with `.venv/bin/python scripts/summ.py results/cv_recent_mu.parquet <reference model>`, and the learning curve with `scripts/lc_summary.py`.
4. Submission archive: `.venv/bin/python scripts/build_submission.py --model "hierlev:infl=0.8:s_de=0.6:s_th=0.5:s_g=1.0:s_grp=1.0:mu0=-1.0" --name v3_mu-1`. It fits the priors on every public benchmark and writes `dist/v3_mu-1.zip`. The submitted archive is in `dist/`. Its sha256 is `9f93e22e153d7ca0945ae96a68a1ca178f47501a3602946e644f73bed5ba2fe3`.
5. Validation and submission use the organizers' [paiec_baseline](https://github.com/aims-foundations/paiec_baseline) repository, which has no license and so is not included. Clone it to `third_party/paiec_baseline`. Then run `third_party/paiec_baseline/check_submission_zip.py dist/v3_mu-1.zip`, and submit with `scripts/submit.py` (it reads `CODABENCH_USERNAME` and `CODABENCH_PASSWORD` from the environment).
6. `scripts/feedback_profile.py` reads the public formative logs of [doctawho42/PAIEC](https://github.com/doctawho42/PAIEC) and [aghasalim/predictive-eval](https://github.com/aghasalim/predictive-eval), cloned under `third_party/`.

Model names in `jobs.txt` use the syntax `base:key=value:...`. `base` is `smooth`, `hier`, `hierlev` or `hierbow`. The keys are the fields of `paie.model.Hyper`, plus `infl` (the level prior width as a multiple of the regression residual sd) and `ridge`.

## Layout

| path | contents |
|---|---|
| `paie/model.py` | the predictor (Laplace fit, prediction, JSON prior) |
| `paie/level.py` | features and the level regression |
| `paie/sim.py` | formative-like runs and budget-major scoring |
| `paie/data.py`, `paie/models_zoo.py` | data loading and model factories for CV |
| `scripts/` | data download, CV, summaries, archive build and submission |
| `submission/model.py` | the archive's entry point |
| `results/` | CV outputs (per-pair Brier, no item data), learning curve, submission log |

## Acknowledgements

The simulator follows the protocol documented in [doctawho42/PAIEC](https://github.com/doctawho42/PAIEC) and the organizers' streaming client. The run shape comes from public formative logs in that repository and in [aghasalim/predictive-eval](https://github.com/aghasalim/predictive-eval). The data is measurement-db by N. Truong, S. T. Truong and S. Koyejo (AIMS Lab, Stanford).

## Use of AI assistants

Claude (Anthropic), working as a coding agent, wrote most of this code and ran the experiments under the author's direction. The author reviewed the method, the results and the text. The competition permits LLM coding assistants and asks for this disclosure.

## License

MIT, see `LICENSE`.
