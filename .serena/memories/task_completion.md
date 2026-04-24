# What to do when a task is completed

## Minimum completion checklist
- Check changed Python files for syntax issues.
- Run `python -m compileall src app.py` after code changes.
- If bot behavior changed, run `python app.py` and smoke-test the affected Telegram flow.
- If provider-specific logic changed, run the relevant standalone integration script when credentials are available:
  - `python test_gpt.py`
  - `python test_groq.py`
  - `python test_github.py`
- Review for secret leakage: this repo contains test scripts that may embed credentials, and `.gitignore` excludes them for that reason.

## Current limits
- No formal lint/format/test automation appears configured in-repo.
- Many checks are integration-dependent and require real API credentials and Telegram bot setup.
- State is in-memory, so restart-related behavior should be considered when changing handlers/runtime/state.