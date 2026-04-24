# Suggested commands for GeminiTelegram (Windows / PowerShell)

## Environment setup
- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`

## Run the bot
- `python app.py`

## Run manual integration scripts
- `python test_gpt.py`
- `python test_groq.py`
- `python test_github.py`

## Basic validation
- `python -m compileall src app.py`

## Useful PowerShell commands
- `Get-ChildItem`
- `Get-ChildItem -Recurse src`
- `Get-Content app.py`
- `git status`
- `git diff`
- `git add <path>`
- `git commit -m "message"`

## Important note
No dedicated formatter/linter/test runner configuration was found in the repository. The practical commands currently available are dependency install, bot startup, compile check, and standalone test scripts.