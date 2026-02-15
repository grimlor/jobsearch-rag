# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly by emailing the maintainer directly rather than opening a public
issue.

## Scope

This project performs browser automation against job board websites. Security
considerations include:

- **Session cookies** — Playwright storage state files contain authentication
  cookies. These are git-ignored by default. Never commit them.
- **Resume data** — `data/resume.md` contains professional background info.
  If you fork this project, replace it with your own and ensure no PII
  (phone, email, address) is included. To prevent Git from tracking your
  local changes to the file, run:

  ```bash
  git update-index --skip-worktree data/resume.md
  ```

  This keeps the committed version in the repo but hides your local edits
  from `git status` and `git add`. To undo:
  `git update-index --no-skip-worktree data/resume.md`
- **Decision history** — `data/decisions/` may contain job preferences that
  reveal your career interests. Also git-ignored.

## Best Practices

- Keep your `.gitignore` intact — it protects sensitive files by default
- Don't share Playwright storage state files
- Don't commit `data/` contents to public repositories
- Review adapter fixtures before committing to ensure they don't contain
  personal data from your authenticated sessions
