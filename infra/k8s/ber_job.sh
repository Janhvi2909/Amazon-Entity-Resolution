#!/usr/bin/env bash
# Launch a detached ber/ pipeline Job on nst-n7 (24 cores, 62 GB).
#   infra/k8s/ber_job.sh <job-name> <code-ts> <cpu-req> <mem-req-Gi> <mem-limit-Gi> <<'EOF'
#   ...shell steps, using run_step <name> <cmd...>...
#   EOF
# Code: /data/ber/code_<ts>/src (PVC). Work: hostPath /var/lib/amlc-ber/work on nst-n7 (680 GB free,
# survives pod restarts on that node). Logs, step status and outputs: /data/ber/logs/<job>/ on the PVC.
# run_step records "<name> EXIT=<rc>" in STEPS.txt, the last 50 log lines on failure, and never
# stops the job: later steps still run.
set -euo pipefail
NAME=$1; TS=$2; CPU=$3; MEMR=$4; MEML=$5
BODY=$(cat)
kubectl apply --request-timeout=60s -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${NAME}
  namespace: amlc-ber-jy
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {app: amlc-ber, job: ${NAME}}
    spec:
      nodeSelector: {kubernetes.io/hostname: nst-n7}
      restartPolicy: OnFailure
      containers:
      - name: run
        image: python:3.12-slim
        command: ["bash", "-c"]
        args:
        - |
          set -u
          export PIP_CACHE_DIR=/data/wheels PYTHONUNBUFFERED=1
          LOG=/data/ber/logs/${NAME}; mkdir -p \$LOG /work
          apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libgomp1 procps >/dev/null 2>&1
          pip install --quiet --cache-dir=/data/wheels -r /data/ber/code_${TS}/requirements.txt 2>&1 | tail -2
          cd /data/ber/code_${TS}/src
          DATA=/data/dataset; WORK=/work; VAL=/data/ber/code_${TS}/src/validate_submission.py
          run_step () {
            local name=\$1; shift
            echo "=== \$name start \$(date -u +%FT%TZ)" | tee -a \$LOG/STEPS.txt
            "\$@" > \$LOG/\$name.log 2>&1; local rc=\$?
            echo "=== \$name EXIT=\$rc \$(date -u +%FT%TZ) container_peak_bytes=\$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)" | tee -a \$LOG/STEPS.txt
            if [ \$rc -ne 0 ]; then tail -50 \$LOG/\$name.log > \$LOG/\$name.FAILED.txt; ps aux --sort=-rss | head -5 >> \$LOG/\$name.FAILED.txt; fi
            for pf in \$WORK/*/PROGRESS.txt; do cp -f \$pf \$LOG/PROGRESS.\$(basename \$(dirname \$pf)).txt 2>/dev/null; done; true
            return 0
          }
          ( while true; do for pf in \$WORK/*/PROGRESS.txt; do cp -f \$pf \$LOG/PROGRESS.\$(basename \$(dirname \$pf)).txt 2>/dev/null; done; sleep 60; done ) &
$(echo "$BODY" | sed 's/^/          /')
          echo "=== JOB DONE \$(date -u +%FT%TZ)" | tee -a \$LOG/STEPS.txt
        resources:
          requests: {cpu: "${CPU}", memory: "${MEMR}Gi", ephemeral-storage: "4Gi"}
          limits: {memory: "${MEML}Gi"}
        volumeMounts:
        - {name: data, mountPath: /data}
        - {name: work, mountPath: /work}
      volumes:
      - name: data
        persistentVolumeClaim: {claimName: amlc-data}
      - name: work
        hostPath: {path: /var/lib/amlc-ber/work, type: DirectoryOrCreate}
EOF
