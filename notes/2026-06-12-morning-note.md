# Morning Note — 2026-06-12

**Tag:** `KOD` · **Project:** `robotic-bearing-pdm` · **Branch:** `claude/morning-note-kod-2pmq8c`

> Daily status briefing for the LSTM-Autoencoder bearing predictive-maintenance project.

---

## TL;DR

The project is in a **polished, portfolio-complete state**. Working tree is clean, no
open PRs or issues, and the docs/dashboard/video assets are all in place. The single
notable gap is **developer-environment reproducibility**: a fresh checkout has no Python
dependencies installed and no automated setup, so the test suite can't run without manual
`pip install`. That's the highest-leverage thing to fix next.

---

## Repository state

| Item | Status |
|---|---|
| Current branch | `claude/morning-note-kod-2pmq8c` |
| Working tree | ✅ Clean — no uncommitted changes |
| Open pull requests | None |
| Open issues | None |
| CI workflows | ⚠️ None configured (no `.github/workflows`) |

### Recent commits (last 8)

- `3ddb5ef` chore: remove old MKV file, MP4 release asset replaces it
- `eec16dd` docs(readme): link video thumbnail to GitHub Release asset
- `b177874` docs(readme): add clickable video thumbnail in header
- `3d87e94` feat: add MP4 explainer video for GitHub preview
- `813aae0` feat: add 90s animated explainer video summarizing project
- `bbc39b2` fix(dashboard): rolling history refresh on rerun, LIVE MODE banner when API online
- `2f2b3dc` test(api): add raw-input normalisation contract test
- `e93e801` fix(api): normalise in /predict, load model config from JSON with tensor fallback

**Read:** recent work has shifted from core ML/API fixes toward **presentation/polish**
(explainer video, README thumbnails, release assets). The functional pipeline appears settled.

---

## Health check

- **Tests:** ⚠️ Could not run in a fresh container — `numpy`, `torch`, and `pytest`
  are not installed, and there is no `.claude/settings.json` SessionStart hook to install
  them automatically. `requirements.txt` pins everything (torch 2.3.0, numpy 1.26.4,
  pytest 8.2.1), so a `pip install -r requirements.txt` is required before the
  `tests/` suite (`test_api.py`, `test_features.py`, `test_model.py`) can execute.
- **Model results** (from `FINDINGS.md`): 123-hour detection lead time, μ+3σ threshold
  = 0.8542, <5% false-positive rate, <20 ms CPU inference — all unchanged.

---

## Suggested focus for today

1. **Reproducible dev setup (highest leverage).** Add a SessionStart hook (or a
   `make setup` / setup script) that runs `pip install -r requirements.txt` so the test
   suite is runnable on a fresh clone. Without this, no automated verification is possible.
2. **CI.** With no `.github/workflows`, nothing guards regressions. A minimal
   `pytest` GitHub Actions workflow would lock in the test fixes from recent commits
   (`2f2b3dc`, `5a6baf7`, `86352bc`, `32f21da`).
3. **Real-data validation.** `FINDINGS.md` notes results are on the *synthetic* dataset —
   swapping in real NASA IMS data remains the standing item for production-grade validation.

---

*Generated as the `KOD` morning note. No code behaviour was changed — this is a status note only.*
