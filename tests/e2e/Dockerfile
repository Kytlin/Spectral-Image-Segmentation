# Browser runner for run_e2e.py: Playwright's image ships Chromium; add the matching Python package.
FROM mcr.microsoft.com/playwright/python:v1.48.0-noble
RUN pip install --no-cache-dir --break-system-packages playwright==1.48.0
