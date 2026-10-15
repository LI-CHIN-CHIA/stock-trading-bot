#!/usr/bin/env bash
# 一鍵部署 Grafana 監控 stack
set -e
cd "$(dirname "$0")/../.."   # project root

echo "=== [1/5] Build trading-exporter image ==="
docker build -t trading-exporter:latest -f Dockerfile.trading-exporter .
echo "  Importing to k3s..."
docker save trading-exporter:latest | sudo k3s ctr images import -

echo "=== [2/5] Apply namespace ==="
kubectl apply -f k8s/monitoring/namespace.yaml

echo "=== [3/5] Deploy Prometheus stack ==="
kubectl apply -f k8s/monitoring/prometheus/rbac.yaml
kubectl apply -f k8s/monitoring/prometheus/configmap.yaml
kubectl apply -f k8s/monitoring/prometheus/deployment.yaml

echo "=== [4/5] Deploy kube-state-metrics + node-exporter ==="
kubectl apply -f k8s/monitoring/kube-state-metrics/deployment.yaml
kubectl apply -f k8s/monitoring/node-exporter/deployment.yaml

echo "=== [5/5] Deploy Grafana + trading-exporter ==="
kubectl apply -f k8s/monitoring/grafana/configmap.yaml
kubectl apply -f k8s/monitoring/grafana/deployment.yaml
kubectl apply -f k8s/monitoring/trading-exporter/deployment.yaml

echo ""
echo "=== 等待 pods 啟動（最多 120s）==="
kubectl rollout status deployment/prometheus      -n monitoring --timeout=120s
kubectl rollout status deployment/grafana         -n monitoring --timeout=120s
kubectl rollout status deployment/trading-exporter -n monitoring --timeout=120s

echo ""
echo "✅ 部署完成！"
echo ""
echo "  開啟 Grafana："
echo "  kubectl port-forward -n monitoring svc/grafana 3000:3000"
echo "  → http://localhost:3000  (admin / trading2026)"
echo ""
echo "  開啟 Prometheus："
echo "  kubectl port-forward -n monitoring svc/prometheus 9090:9090"
echo "  → http://localhost:9090"
