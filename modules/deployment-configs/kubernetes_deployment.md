# Kubernetes Deployment Configuration

## 1. Namespace and RBAC (`00-namespace.yaml`)

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: fiber-inspection
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fiber-inspector
  namespace: fiber-inspection
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: fiber-inspector-role
  namespace: fiber-inspection
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: fiber-inspector-binding
  namespace: fiber-inspection
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: fiber-inspector-role
subjects:
- kind: ServiceAccount
  name: fiber-inspector
  namespace: fiber-inspection
```

## 2. ConfigMap (`01-configmap.yaml`)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fiber-config
  namespace: fiber-inspection
data:
  config.json: |
    {
      "model_path": "/models/fiber_model.pth",
      "redis_host": "redis-service",
      "redis_port": "6379",
      "project_id": "your-project-id",
      "topic_name": "fiber-inspection",
      "alert_webhook": "https://alerts.your-domain.com/webhook",
      "ws_host": "0.0.0.0",
      "ws_port": "8765",
      "threads": 8,
      "processes": 4,
      "streams": []
    }
  
  nginx.conf: |
    upstream websocket {
        server localhost:8765;
    }
    
    upstream api {
        server localhost:8000;
    }
    
    server {
        listen 80;
        server_name _;
        
        location /ws {
            proxy_pass http://websocket;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 86400;
        }
        
        location /api {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location / {
            root /usr/share/nginx/html;
            try_files $uri $uri/ /index.html;
        }
    }
```

## 3. Secrets (`02-secrets.yaml`)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: fiber-secrets
  namespace: fiber-inspection
type: Opaque
stringData:
  firebase-key.json: |
    {
      "type": "service_account",
      "project_id": "your-project-id",
      "private_key_id": "your-key-id",
      "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    }
  
  gcp-key.json: |
    {
      "type": "service_account",
      "project_id": "your-project-id",
      "private_key_id": "your-key-id",
      "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    }
```

## 4. Redis Deployment (`03-redis.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: fiber-inspection
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: fiber-inspection
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: fiber-inspection
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

## 5. Main Processing Deployment (`04-processor.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fiber-processor
  namespace: fiber-inspection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fiber-processor
  template:
    metadata:
      labels:
        app: fiber-processor
    spec:
      serviceAccountName: fiber-inspector
      containers:
      - name: processor
        image: gcr.io/your-project-id/fiber-processor:latest
        ports:
        - containerPort: 8765  # WebSocket
        - containerPort: 8000  # Prometheus metrics
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /secrets/gcp-key.json
        - name: FIREBASE_CONFIG
          value: /secrets/firebase-key.json
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1  # Request GPU if available
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config
          mountPath: /config
        - name: secrets
          mountPath: /secrets
        - name: models
          mountPath: /models
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: config
        configMap:
          name: fiber-config
      - name: secrets
        secret:
          secretName: fiber-secrets
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: processor-service
  namespace: fiber-inspection
spec:
  selector:
    app: fiber-processor
  ports:
  - name: websocket
    port: 8765
    targetPort: 8765
  - name: metrics
    port: 8000
    targetPort: 8000
```

## 6. Feature Extraction Service (`05-feature-service.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-extractor
  namespace: fiber-inspection
spec:
  replicas: 2
  selector:
    matchLabels:
      app: feature-extractor
  template:
    metadata:
      labels:
        app: feature-extractor
    spec:
      containers:
      - name: extractor
        image: gcr.io/your-project-id/feature-extractor:latest
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080"
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /secrets/gcp-key.json
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: secrets
          mountPath: /secrets
      volumes:
      - name: secrets
        secret:
          secretName: fiber-secrets
---
apiVersion: v1
kind: Service
metadata:
  name: feature-service
  namespace: fiber-inspection
spec:
  selector:
    app: feature-extractor
  ports:
  - port: 8080
    targetPort: 8080
```

## 7. Dashboard Deployment (`06-dashboard.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fiber-dashboard
  namespace: fiber-inspection
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fiber-dashboard
  template:
    metadata:
      labels:
        app: fiber-dashboard
    spec:
      containers:
      - name: dashboard
        image: gcr.io/your-project-id/fiber-dashboard:latest
        ports:
        - containerPort: 80
        env:
        - name: REACT_APP_WS_URL
          value: "wss://fiber-inspection.your-domain.com/ws"
        - name: REACT_APP_API_URL
          value: "https://fiber-inspection.your-domain.com/api"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
  namespace: fiber-inspection
spec:
  selector:
    app: fiber-dashboard
  ports:
  - port: 80
    targetPort: 80
```

## 8. Ingress Configuration (`07-ingress.yaml`)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fiber-ingress
  namespace: fiber-inspection
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/websocket-services: "processor-service"
spec:
  tls:
  - hosts:
    - fiber-inspection.your-domain.com
    secretName: fiber-tls
  rules:
  - host: fiber-inspection.your-domain.com
    http:
      paths:
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: processor-service
            port:
              number: 8765
      - path: /api/features
        pathType: Prefix
        backend:
          service:
            name: feature-service
            port:
              number: 8080
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dashboard-service
            port:
              number: 80
```

## 9. Horizontal Pod Autoscaler (`08-hpa.yaml`)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: processor-hpa
  namespace: fiber-inspection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fiber-processor
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
  - type: Pods
    pods:
      metric:
        name: fiber_queue_size
      target:
        type: AverageValue
        averageValue: "100"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: feature-hpa
  namespace: fiber-inspection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: feature-extractor
  minReplicas: 2
  maxReplicas: 6
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
```

## 10. Monitoring Stack (`09-monitoring.yaml`)

```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: fiber-metrics
  namespace: fiber-inspection
spec:
  selector:
    matchLabels:
      app: fiber-processor
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: fiber-alerts
  namespace: fiber-inspection
spec:
  groups:
  - name: fiber.rules
    interval: 30s
    rules:
    - alert: HighAnomalyRate
      expr: rate(fiber_anomalies_detected_total[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High anomaly detection rate"
        description: "Anomaly rate is {{ $value }} per second"
    
    - alert: ProcessingQueueFull
      expr: fiber_queue_size > 900
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "Processing queue is almost full"
        description: "Queue size is {{ $value }}"
    
    - alert: LowQualityScore
      expr: avg(fiber_quality_score) < 70
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Low average quality score"
        description: "Average quality is {{ $value }}%"
```

## 11. Deployment Script (`deploy.sh`)

```bash
#!/bin/bash
# Kubernetes deployment script for Fiber Inspection System

set -e

NAMESPACE="fiber-inspection"
PROJECT_ID="your-project-id"
CLUSTER_NAME="fiber-cluster"
REGION="us-central1"

echo "🚀 Deploying Fiber Inspection System to Kubernetes..."

# Build and push Docker images
echo "📦 Building Docker images..."
docker build -t gcr.io/$PROJECT_ID/fiber-processor:latest -f Dockerfile.processor .
docker build -t gcr.io/$PROJECT_ID/feature-extractor:latest -f Dockerfile.feature .
docker build -t gcr.io/$PROJECT_ID/fiber-dashboard:latest -f Dockerfile.dashboard .

echo "📤 Pushing images to GCR..."
docker push gcr.io/$PROJECT_ID/fiber-processor:latest
docker push gcr.io/$PROJECT_ID/feature-extractor:latest
docker push gcr.io/$PROJECT_ID/fiber-dashboard:latest

# Connect to cluster
echo "🔗 Connecting to GKE cluster..."
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID

# Apply configurations
echo "⚙️ Applying Kubernetes configurations..."
kubectl apply -f 00-namespace.yaml
kubectl apply -f 01-configmap.yaml
kubectl apply -f 02-secrets.yaml
kubectl apply -f 03-redis.yaml
kubectl apply -f 04-processor.yaml
kubectl apply -f 05-feature-service.yaml
kubectl apply -f 06-dashboard.yaml
kubectl apply -f 07-ingress.yaml
kubectl apply -f 08-hpa.yaml
kubectl apply -f 09-monitoring.yaml

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment --all -n $NAMESPACE

# Get service endpoints
echo "✅ Deployment complete!"
echo "📍 Service endpoints:"
kubectl get ingress -n $NAMESPACE

echo "📊 To view the dashboard, run:"
echo "   kubectl port-forward -n $NAMESPACE svc/dashboard-service 8080:80"
echo "   Then open http://localhost:8080"

echo "📈 To view metrics:"
echo "   kubectl port-forward -n $NAMESPACE svc/processor-service 9090:8000"
echo "   Then open http://localhost:9090/metrics"
```

## 12. Grafana Dashboard (`grafana-dashboard.json`)

```json
{
  "dashboard": {
    "title": "Fiber Optic Inspection Metrics",
    "panels": [
      {
        "title": "Frames Processed Rate",
        "targets": [
          {
            "expr": "rate(fiber_frames_processed_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Anomaly Detection Rate",
        "targets": [
          {
            "expr": "rate(fiber_anomalies_detected_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Processing Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, fiber_processing_duration_seconds_bucket)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Streams",
        "targets": [
          {
            "expr": "fiber_active_streams"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Queue Size",
        "targets": [
          {
            "expr": "fiber_queue_size"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

## 13. Backup and Disaster Recovery (`10-backup.yaml`)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: fiber-backup
  namespace: fiber-inspection
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: gcr.io/your-project-id/fiber-backup:latest
            env:
            - name: BACKUP_BUCKET
              value: gs://your-backup-bucket
            - name: NAMESPACE
              value: fiber-inspection
            command:
            - /bin/bash
            - -c
            - |
              # Backup Redis
              kubectl exec -n $NAMESPACE redis-0 -- redis-cli BGSAVE
              sleep 10
              kubectl cp $NAMESPACE/redis-0:/data/dump.rdb /tmp/redis-backup.rdb
              gsutil cp /tmp/redis-backup.rdb $BACKUP_BUCKET/redis/$(date +%Y%m%d)/dump.rdb
              
              # Backup Firestore
              gcloud firestore export $BACKUP_BUCKET/firestore/$(date +%Y%m%d)
              
              # Backup models
              gsutil -m cp -r /models/* $BACKUP_BUCKET/models/$(date +%Y%m%d)/
          restartPolicy: OnFailure
```