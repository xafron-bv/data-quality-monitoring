# Deployment Examples

This document provides example configurations for deploying the Data Quality Detection System in various environments.

> **Note**: The system is a CLI batch tool. Integrations wrap the CLI; there is no built-in server.

## Current Deployment Method

Run as a batch process from Python:

```bash
python main.py single-demo --data-file data.csv
```

## Example 1: Simple Batch Orchestration (Python)

```python
# run_detection.py
import os
import subprocess
from datetime import datetime

PROJECT_DIR = "/opt/data-quality-detection"
DATA_DIR = "/data/incoming"
OUTPUT_DIR = "/data/results"
LOG_DIR = "/var/log/detection"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(OUTPUT_DIR, exist_ok=True)
result = subprocess.run([
    "python", os.path.join(PROJECT_DIR, "main.py"), "single-demo",
    "--data-file", os.path.join(DATA_DIR, "latest.csv"),
    "--output-dir", os.path.join(OUTPUT_DIR, ts),
    "--enable-validation", "--enable-pattern", "--enable-ml",
], capture_output=True, text=True)
open(os.path.join(LOG_DIR, f"detection_{ts}.log"), "w").write(result.stdout + "\n" + result.stderr)
```

## Example 2: Airflow DAG (Python)

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess

def run_detection(ds, **_):
    subprocess.run([
        "python", "/opt/data-quality-detection/main.py", "single-demo",
        "--data-file", "/data/daily_export.csv",
        "--output-dir", f"/results/{ds}",
        "--enable-validation", "--enable-pattern", "--enable-ml",
    ])

with DAG(
    dag_id="data_quality_detection",
    default_args={
        "owner": "data-team",
        "depends_on_past": False,
        "start_date": datetime(2024, 1, 1),
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule_interval="0 2 * * *",
    catchup=False,
) as dag:
    task = PythonOperator(task_id="run_detection", python_callable=run_detection)
```

## Example 3: File Watcher (Python)

```python
# monitor.py
import time
import subprocess
from pathlib import Path
from datetime import datetime

WATCH_DIR = Path("/data/incoming")
OUTPUT_DIR = Path("/data/results")

seen = set()
while True:
    for path in WATCH_DIR.glob("*.csv"):
        if path in seen:
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / ts
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python", "main.py", "single-demo",
            "--data-file", str(path),
            "--output-dir", str(out),
        ])
        seen.add(path)
    time.sleep(60)
```

## Production Deployment Checklist

- Environment
  - Python 3.8+
  - Virtual environment
  - Dependencies installed
  - GPU drivers (if using ML/LLM)
- Configuration
  - Brand configurations created
  - Thresholds tuned for your data
  - Output directories configured
  - Logging configured
- Data Pipeline
  - Input data format verified
  - Output location accessible
  - Error handling in place
  - Monitoring/alerting configured
- Performance
  - Batch size optimized
  - Memory limits considered
  - GPU allocation configured
  - Parallel processing tuned
- Security
  - File permissions set correctly
  - Sensitive data handling reviewed
  - Network access restricted
  - Audit logging enabled

## Notes
- Adapt examples to your infrastructure
- Test in staging before production
- Monitor resource usage and adjust as needed
- Consider data volume and processing frequency