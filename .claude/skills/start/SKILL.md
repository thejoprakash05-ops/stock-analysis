---
description: Start the Stock Analysis Streamlit app on http://localhost:8501
tools: Bash
---

Start the Stock Analysis Streamlit app.

## Steps

1. Check if port 8501 is already occupied:
   ```
   netstat -ano | findstr ":8501"
   ```
   If any line shows `LISTENING` or `ESTABLISHED`, the app is already running — report the URL and stop.

2. Launch Streamlit in the background:
   ```
   "C:\Users\jahna\AppData\Roaming\Python\Python314\Scripts\streamlit.exe" run "c:\Users\jahna\StockAnalysis\app.py" --server.port 8501 --server.headless true
   ```
   Use `run_in_background: true`.

3. Poll until the health endpoint responds 200 (up to 20 s):
   ```
   until curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/healthz | grep -q "200"; do sleep 2; done
   ```

4. Report: "App is running at **http://localhost:8501**"
