---
description: Commit all changes and push the Stock Analysis app to GitHub
tools: Bash
---

Stage, commit, and push all changes in c:\Users\jahna\StockAnalysis to GitHub.

## Steps

1. Change to the project directory:
   ```
   cd "c:\Users\jahna\StockAnalysis"
   ```

2. Initialize git if not already a repo:
   ```
   git status
   ```
   If `fatal: not a git repository`, run:
   ```
   git init
   git branch -M main
   ```

3. Check whether a remote named `origin` exists:
   ```
   git remote -v
   ```
   - If no remote is set, ask the user: "Please provide your GitHub repository URL (e.g. https://github.com/user/repo.git):"
     Then add it: `git remote add origin <url>`

4. Check for uncommitted changes:
   ```
   git status --short
   ```
   If the working tree is clean, report "Nothing to push — working tree is clean." and stop.

5. Stage all changes:
   ```
   git add app.py requirements.txt run.bat .gitignore
   ```
   (Exclude screenshots and cache — .gitignore handles those.)

6. Inspect the diff to write an accurate commit message:
   ```
   git diff --cached --stat
   ```
   Draft a concise one-line message that describes what changed (e.g. "Add 3-stock comparison view with AI analysis").

7. Commit:
   ```
   git commit -m "<message>"
   ```

8. Push:
   ```
   git push -u origin main
   ```

9. Report the result and the GitHub URL.
