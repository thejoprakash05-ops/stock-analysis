---
description: Stop the Stock Analysis Streamlit app running on port 8501
tools: Bash
---

Stop the running Streamlit app on port 8501.

## Steps

1. Find the PID of the process listening on port 8501:
   ```
   netstat -ano | findstr ":8501"
   ```
   Extract the PID from the rightmost column of any `LISTENING` line.

2. If no process is found, report "No app is running on port 8501" and stop.

3. Kill the process:
   ```
   taskkill /F /PID <pid>
   ```
   If there are multiple PIDs (unlikely), kill each one.

4. Confirm port 8501 is now free:
   ```
   netstat -ano | findstr ":8501"
   ```
   Report "App stopped." (or "Port 8501 is now free.")
