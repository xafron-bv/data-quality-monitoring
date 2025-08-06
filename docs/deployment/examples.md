# Deployment Examples

This document provides example configurations for deploying the Data Quality Detection System in various environments.

> **⚠️ IMPORTANT NOTE**: The examples in this document are provided as reference implementations. The current system is a command-line batch processing tool without built-in support for Docker, Kubernetes, or API endpoints. These examples show how you might deploy the system in production environments with custom wrapper scripts and configurations.

## Current Deployment Method

The system is designed to run as a batch process:

```bash
# Basic execution
python main.py single-demo --data-file data.csv

# Scheduled execution with cron
0 2 * * * cd /path/to/project && /path/to/venv/bin/python main.py single-demo --data-file /data/daily.csv --output-dir /results/$(date +\%Y\%m\%d)
```

## Example Configurations

### Example 1: Simple Batch Processing Script

**File: `run_detection.sh`** (EXAMPLE)
```bash
#!/bin/bash
# Example wrapper script for production deployment

# Configuration
PROJECT_DIR="/opt/data-quality-detection"
VENV_PATH="$PROJECT_DIR/venv"
DATA_DIR="/data/incoming"
OUTPUT_DIR="/data/results"
LOG_DIR="/var/log/detection"

# Activate virtual environment
source $VENV_PATH/bin/activate

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$OUTPUT_DIR/$TIMESTAMP"
mkdir -p $RESULT_DIR

# Run detection
cd $PROJECT_DIR
python main.py single-demo \
    --data-file $DATA_DIR/latest.csv \
    --output-dir $RESULT_DIR \
    --enable-validation \
    --enable-pattern \
    --enable-ml \
    2>&1 | tee $LOG_DIR/detection_$TIMESTAMP.log

# Check exit status
if [ $? -eq 0 ]; then
    echo "Detection completed successfully"
    # Optional: trigger downstream processes
else
    echo "Detection failed"
    # Optional: send alert
fi
```

### Example 2: Docker Configuration (NOT IMPLEMENTED)

**File: `Dockerfile`** (EXAMPLE - would need to be created)
```dockerfile
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/results /app/logs /app/data

# Entry point
ENTRYPOINT ["python", "main.py"]
```

**File: `docker-compose.yml`** (EXAMPLE)
```yaml
version: '3.8'

services:
  detection:
    build: .
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./brand_configs:/app/brand_configs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: single-demo --data-file /app/data/input.csv
```

### Example 3: Kubernetes CronJob (NOT IMPLEMENTED)

**File: `detection-cronjob.yaml`** (EXAMPLE)
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: data-quality-detection
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: detection
            image: your-registry/data-quality-detection:latest
            command:
              - python
              - main.py
              - single-demo
              - --data-file
              - /data/daily.csv
              - --output-dir
              - /results
            volumeMounts:
            - name: data
              mountPath: /data
            - name: results
              mountPath: /results
            - name: config
              mountPath: /app/brand_configs
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
              limits:
                memory: "8Gi"
                cpu: "4"
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: detection-data-pvc
          - name: results
            persistentVolumeClaim:
              claimName: detection-results-pvc
          - name: config
            configMap:
              name: brand-configs
          restartPolicy: OnFailure
```

### Example 4: Airflow DAG (INTEGRATION EXAMPLE)

**File: `detection_dag.py`** (EXAMPLE)
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_quality_detection',
    default_args=default_args,
    description='Daily data quality detection',
    schedule_interval='0 2 * * *',
    catchup=False,
)

# Task 1: Prepare data
prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command='''
    # Export data from database to CSV
    psql -h $DB_HOST -U $DB_USER -d $DB_NAME \
        -c "COPY (SELECT * FROM products) TO '/data/daily_export.csv' CSV HEADER"
    ''',
    dag=dag,
)

# Task 2: Run detection
run_detection = BashOperator(
    task_id='run_detection',
    bash_command='''
    cd /opt/data-quality-detection
    source venv/bin/activate
    python main.py single-demo \
        --data-file /data/daily_export.csv \
        --output-dir /results/{{ ds }} \
        --enable-validation \
        --enable-pattern \
        --enable-ml
    ''',
    dag=dag,
)

# Task 3: Process results
def process_results(**context):
    import json
    import pandas as pd
    
    date = context['ds']
    with open(f'/results/{date}/report.json', 'r') as f:
        report = json.load(f)
    
    # Extract metrics
    total_anomalies = report['summary']['total_anomalies']
    
    # Alert if threshold exceeded
    if total_anomalies > 100:
        context['task_instance'].xcom_push(key='alert', value=True)

process_results_task = PythonOperator(
    task_id='process_results',
    python_callable=process_results,
    dag=dag,
)

# Define task dependencies
prepare_data >> run_detection >> process_results_task
```

### Example 5: Systemd Service (LINUX DEPLOYMENT)

**File: `/etc/systemd/system/detection-monitor.service`** (EXAMPLE)
```ini
[Unit]
Description=Data Quality Detection Monitor
After=network.target

[Service]
Type=simple
User=detection
Group=detection
WorkingDirectory=/opt/data-quality-detection
Environment="PATH=/opt/data-quality-detection/venv/bin"
ExecStart=/opt/data-quality-detection/venv/bin/python /opt/data-quality-detection/monitor.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**File: `monitor.py`** (EXAMPLE - would need to be created)
```python
#!/usr/bin/env python
"""Example monitoring script that watches for new files and runs detection."""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

WATCH_DIR = Path("/data/incoming")
OUTPUT_DIR = Path("/data/results")
PROCESSED_DIR = Path("/data/processed")

def process_file(filepath):
    """Run detection on a single file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "main.py", "single-demo",
        "--data-file", str(filepath),
        "--output-dir", str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Move processed file
        processed_path = PROCESSED_DIR / f"{filepath.stem}_{timestamp}{filepath.suffix}"
        filepath.rename(processed_path)
        print(f"Successfully processed {filepath}")
    else:
        print(f"Error processing {filepath}: {result.stderr}")

def monitor():
    """Monitor directory for new CSV files."""
    processed_files = set()
    
    while True:
        for filepath in WATCH_DIR.glob("*.csv"):
            if filepath not in processed_files:
                print(f"New file detected: {filepath}")
                process_file(filepath)
                processed_files.add(filepath)
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor()
```

## Production Deployment Checklist

When deploying to production, consider:

1. **Environment Setup**
   - [ ] Python 3.8+ installed
   - [ ] Virtual environment created
   - [ ] All dependencies installed
   - [ ] GPU drivers (if using ML/LLM)

2. **Configuration**
   - [ ] Brand configurations created
   - [ ] Thresholds tuned for your data
   - [ ] Output directories configured
   - [ ] Logging configured

3. **Data Pipeline Integration**
   - [ ] Input data format verified
   - [ ] Output location accessible
   - [ ] Error handling in place
   - [ ] Monitoring/alerting configured

4. **Performance**
   - [ ] Batch size optimized
   - [ ] Memory limits set
   - [ ] GPU allocation configured
   - [ ] Parallel processing tuned

5. **Security**
   - [ ] File permissions set correctly
   - [ ] Sensitive data handling reviewed
   - [ ] Network access restricted
   - [ ] Audit logging enabled

## Notes on Examples

- These examples show common deployment patterns
- Adapt them to your specific infrastructure
- Test thoroughly in staging before production
- Monitor resource usage and adjust as needed
- Consider data volume and processing frequency

Remember: The core system is a Python application that processes CSV files. All deployment examples are wrappers around this basic functionality.