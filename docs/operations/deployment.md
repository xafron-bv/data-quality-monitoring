# Deployment Guide

This guide covers deployment options and best practices for the Data Quality Detection System in production environments.

## Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx/HAProxy]
    end
    
    subgraph "Application Tier"
        A1[Detection Service 1]
        A2[Detection Service 2]
        A3[Detection Service N]
    end
    
    subgraph "Processing Tier"
        W1[Worker 1<br/>ML/LLM Processing]
        W2[Worker 2<br/>ML/LLM Processing]
        W3[Worker N<br/>ML/LLM Processing]
    end
    
    subgraph "Storage Tier"
        D1[(Input Data)]
        D2[(Model Storage)]
        D3[(Results Storage)]
        D4[(Configuration)]
    end
    
    subgraph "Monitoring"
        M1[Metrics]
        M2[Logs]
        M3[Alerts]
    end
    
    LB --> A1 & A2 & A3
    A1 & A2 & A3 --> W1 & W2 & W3
    W1 & W2 & W3 --> D1 & D2 & D3 & D4
    A1 & A2 & A3 --> M1 & M2 & M3
```

## Deployment Options

### 1. Standalone Deployment

Simple deployment for small-scale usage.

#### System Requirements
- CPU: 4+ cores
- RAM: 16GB minimum (32GB recommended)
- Storage: 50GB+ SSD
- GPU: Optional (NVIDIA CUDA-capable for ML acceleration)

#### Installation Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd detection-system

# 2. Create virtual environment
python3.8 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Initialize models
python scripts/download_models.py

# 6. Run service
python main.py serve --port 8080
```

### 2. Docker Deployment

Containerized deployment for better isolation and portability.

#### Dockerfile

```dockerfile
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run application
CMD ["python", "main.py", "serve", "--port", "8080"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  detection-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - MODEL_PATH=/models
      - DATA_PATH=/data
    volumes:
      - ./models:/models:ro
      - ./data:/data
      - ./results:/results
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  worker:
    build: .
    command: python main.py worker
    environment:
      - ENVIRONMENT=production
      - DEVICE=cuda
    volumes:
      - ./models:/models:ro
      - ./data:/data
      - ./results:/results
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### 3. Kubernetes Deployment

Enterprise-grade deployment with auto-scaling and high availability.

#### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: detection-system
  labels:
    app: detection-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: detection-system
  template:
    metadata:
      labels:
        app: detection-system
    spec:
      containers:
      - name: detection-api
        image: detection-system:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: MODEL_PATH
          value: "/models"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: config
          mountPath: /config
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: config
        configMap:
          name: detection-config
```

#### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: detection-service
spec:
  selector:
    app: detection-system
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: detection-system
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 4. Cloud Deployment

#### AWS Deployment

Using AWS ECS with Fargate:

```yaml
# task-definition.json
{
  "family": "detection-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "detection-api",
      "image": "your-ecr-repo/detection-system:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/detection-system",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: detection-system
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT-ID/detection-system
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: ENVIRONMENT
          value: production
```

## Production Configuration

### Environment Variables

```bash
# Application settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# API settings
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4

# Model settings
MODEL_PATH=/models
MODEL_CACHE_SIZE=10
MODEL_DEVICE=cuda

# Data settings
DATA_PATH=/data
MAX_BATCH_SIZE=1000
MAX_FILE_SIZE=1GB

# Performance settings
ENABLE_CACHE=true
CACHE_TTL=3600
PARALLEL_WORKERS=8

# Security settings
API_KEY_REQUIRED=true
RATE_LIMIT=100
CORS_ORIGINS=https://your-domain.com

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
TRACING_ENABLED=true
```

### Nginx Configuration

```nginx
upstream detection_backend {
    least_conn;
    server detection1:8080 weight=1;
    server detection2:8080 weight=1;
    server detection3:8080 weight=1;
}

server {
    listen 80;
    server_name api.detection.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.detection.example.com;

    ssl_certificate /etc/ssl/certs/detection.crt;
    ssl_certificate_key /etc/ssl/private/detection.key;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Request size limits
    client_max_body_size 100M;
    client_body_timeout 300s;

    # Compression
    gzip on;
    gzip_types application/json text/plain;

    location / {
        proxy_pass http://detection_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        access_log off;
        proxy_pass http://detection_backend/health;
    }
}
```

## Database Setup

### PostgreSQL for Results Storage

```sql
-- Create database
CREATE DATABASE detection_system;

-- Create results table
CREATE TABLE detection_results (
    id SERIAL PRIMARY KEY,
    job_id UUID NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_file VARCHAR(255),
    field_name VARCHAR(100),
    detection_method VARCHAR(50),
    total_records INTEGER,
    errors_detected INTEGER,
    confidence_scores JSONB,
    detailed_results JSONB,
    processing_time FLOAT,
    INDEX idx_job_id (job_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_field_name (field_name)
);

-- Create metrics table
CREATE TABLE detection_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    labels JSONB,
    INDEX idx_metric_timestamp (timestamp, metric_name)
);
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'detection-system'
    static_configs:
      - targets: ['detection1:9090', 'detection2:9090', 'detection3:9090']
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node1:9100', 'node2:9100', 'node3:9100']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Detection System Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Detection Performance",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(detection_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(detection_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Security Considerations

### API Authentication

```python
# api_auth.py
from functools import wraps
from flask import request, jsonify
import jwt

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 403
            
        return f(*args, **kwargs)
    return decorated_function
```

### Rate Limiting

```python
# rate_limit.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route('/api/detect')
@limiter.limit("5 per minute")
def detect():
    pass
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" /models

# Backup configuration
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" /config

# Backup database
pg_dump detection_system > "$BACKUP_DIR/database_$DATE.sql"

# Upload to S3
aws s3 cp "$BACKUP_DIR/" s3://backup-bucket/detection-system/ --recursive

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -type f -mtime +7 -delete
```

## Performance Optimization

### Model Caching

```python
# model_cache.py
import functools
import time

class ModelCache:
    def __init__(self, max_size=10, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())
```

### Connection Pooling

```python
# db_pool.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/detection',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Limit model cache size
   export MODEL_CACHE_SIZE=5
   ```

2. **Slow Response Times**
   ```bash
   # Enable profiling
   export PROFILING=true
   
   # Check slow queries
   tail -f logs/slow_queries.log
   ```

3. **Model Loading Failures**
   ```bash
   # Verify model files
   ls -la /models/
   
   # Check permissions
   chmod -R 755 /models
   ```

## Health Checks

### Endpoints

```python
@app.route('/health')
def health():
    """Basic health check"""
    return {'status': 'ok'}, 200

@app.route('/ready')
def ready():
    """Readiness check"""
    checks = {
        'database': check_database(),
        'models': check_models_loaded(),
        'storage': check_storage_access()
    }
    
    if all(checks.values()):
        return {'status': 'ready', 'checks': checks}, 200
    else:
        return {'status': 'not ready', 'checks': checks}, 503
```

## Next Steps

- Configure [Performance Tuning](performance.md)
- Set up [Monitoring](monitoring.md)
- Review [Troubleshooting Guide](troubleshooting.md)