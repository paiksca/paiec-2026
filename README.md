# PAIEC 2026: a benchmark level prior learned from 182 public benchmarks

This repository holds Samuel Paik-Heintz's entry to the [NeurIPS 2026 Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition) (Codabench 17828). The author is at the Institute for Systems Biology.

The predictor `predict(input, labeled)` returns the probability that an AI system (the subject) answers a benchmark item correctly. The benchmark is absent from the public data, and the predictor sees 0 to 31 revealed labels. It is a hierarchical logistic model. It refits its posterior from `labeled` on every call:

    eta = mu_b + theta_s + delta_sb + g + u_group + e_i

- `mu_b` is the benchmark level. A ridge regression trained on the public pool predicts its prior mean from features of the item text and the subject's harness.
- `theta_s` is the subject standing, keyed on the canonical model name and learned across benchmarks. For an unseen name, a regression on release date, provider, harness and reasoning effort gives the prior mean.
- `delta_sb` is the deviation of the subject-benchmark pair.
- `g` is a level shift that every benchmark in a run shares.
- `u_group` holds item_features group effects.
- `e_i` is the item residual, which the model integrates out.

The submitted archive needs numpy only.

## Results

The evaluation is grouped 8-fold cross-validation over benchmarks. The simulated runs match the shape of 55 public formative feedback logs. Evaluation pairs have subjects released in 2025 or later. ALC is averaged per benchmark, and the SEs of paired differences are clustered by benchmark.

| model | main regime (64 benchmarks) | hard regime (19 benchmarks under 40% accuracy) |
|---|---|---|
| smoothed label mean | 0.2125 | |
| hierarchical, constant level prior | 0.2106 | |
| hierarchical, feature-based level prior | 0.2078 | 0.2029 |
| v2: adds a run shift and item_features groups | 0.2076 | 0.1985 |
| v3: lowers the level centre by 1.0 (submitted) | 0.2108 | 0.1887 |

I wrote the selection rule before the comparison: minimize the mean ALC of the two regimes. The rule picks v3. The feature-based level prior hurts when fitted on 5 to 20 training benchmarks and helps from about 80 (`results/learning_curve.csv`). `results/submissions.tsv` lists the repeated platform draws.

## Reproduction

The code runs on Python 3.11 and a laptop CPU. Every step uses two processes.

1. Install with `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
2. Accept the gated terms of [aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db) with your own Hugging Face account and export `HF_TOKEN`. Then run `.venv/bin/python scripts/fetch_data.py` and `.venv/bin/python scripts/consolidate.py`. The scripts pin revision `54074453da7b9911be0f48e4c560be547bc84d21` and write about 200 MB to `data/consolidated/`. You can delete `data/hf/` afterwards. This repository does not redistribute the data.
3. `.venv/bin/python scripts/local_run.py` runs the 30 cross-validation commands in `scripts/jobs.txt` and skips outputs that exist. They take a few hours on two cores. `scripts/summ.py <result> <reference model>` summarizes one result, and `scripts/lc_summary.py` summarizes the learning curve.
4. `.venv/bin/python scripts/build_submission.py --model "hierlev:infl=0.8:s_de=0.6:s_th=0.5:s_g=1.0:s_grp=1.0:mu0=-1.0" --name v3_mu-1` fits the priors on every public benchmark and writes `dist/v3_mu-1.zip`. The submitted archive is in `dist/`, with sha256 `9f93e22e153d7ca0945ae96a68a1ca178f47501a3602946e644f73bed5ba2fe3`.
5. Validation and submission need the organizers' [paiec_baseline](https://github.com/aims-foundations/paiec_baseline) repository. It has no license, so it is not included. Clone it to `third_party/paiec_baseline` and run `third_party/paiec_baseline/check_submission_zip.py dist/v3_mu-1.zip`. `scripts/submit.py` submits an archive and reads `CODABENCH_USERNAME` and `CODABENCH_PASSWORD` from the environment.
6. `scripts/feedback_profile.py` reads the public formative logs of [doctawho42/PAIEC](https://github.com/doctawho42/PAIEC) and [aghasalim/predictive-eval](https://github.com/aghasalim/predictive-eval). Clone both under `third_party/`.

A model name in `jobs.txt` has the form `base:key=value:...`. The base is `smooth`, `hier`, `hierlev` or `hierbow`. The keys are the fields of `paie.model.Hyper`, plus `infl` and `ridge`. `infl` sets the level prior's width as a multiple of the regression's residual sd.

## Layout

| path | contents |
|---|---|
| `paie/model.py` | the predictor: Laplace fit, prediction and JSON prior |
| `paie/level.py` | features and the level regression |
| `paie/sim.py` | formative-shaped runs and budget-major scoring |
| `paie/data.py`, `paie/models_zoo.py` | data loading and model factories for cross-validation |
| `scripts/` | data download, cross-validation, summaries, archive build and submission |
| `submission/model.py` | the archive's entry point |
| `results/` | per-pair Brier from cross-validation (no item data), the learning curve and the submission log |

## Acknowledgements

The simulator follows the protocol documented in [doctawho42/PAIEC](https://github.com/doctawho42/PAIEC) and in the organizers' streaming client. The run shape comes from public formative logs in that repository and in [aghasalim/predictive-eval](https://github.com/aghasalim/predictive-eval). The data is measurement-db by N. Truong, S. T. Truong and S. Koyejo (AIMS Lab, Stanford).

## Use of AI assistants

Claude (Anthropic), working as a coding agent, wrote most of this code and ran the experiments under the author's direction. The author reviewed the method, the results and the text. The competition permits LLM coding assistants and asks entrants to disclose them.

## License

MIT. See `LICENSE`.
