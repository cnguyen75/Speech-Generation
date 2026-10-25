# Working in this repository

Follow `CODING-RULES.md` in this folder for all code, comments, and documentation. Run
`python3 rules_check.py .` and the formatters named in `rules_check.py` before every commit;
a commit with findings is not ready.

Commit as Colin Nguyen <26684223+cnguyen75@users.noreply.github.com>. Never use any other
identity. Do not push unless asked.

Nothing here may name an employer's internal systems, customers, colleagues, vendors, or
metrics. No credentials, tokens, or private paths in any file, ever.

<!-- Repo-specific notes below: what is private, how to build, what the tests cover. -->

## Repo notes

- Run: `uv sync`, then the "Reproduce" steps in `README.md` (`scripts/prepare_data.py`,
  `scripts/overnight.sh` for the ~10 h study, then the export and figure scripts).
- Tests: `uv run pytest tests` (synthetic data only, no model download). Run it, plus
  `ruff check . && ruff format --check .` and `python3 rules_check.py .`, before every commit.
- Data: `data/` (LibriTTS-R audio, CC BY 4.0), `runs/` (checkpoints and logs), and `pretrained/`
  (SpeechBrain models) are gitignored and downloaded at run time. `results/` holds the study's
  numbers, figures, and showcase audio; it is the record of the run from commit `af156b3` and is
  not edited for style. The reader is identified only by corpus speaker id.
- `v1-course-project/` is the archived course submission and is not edited.
