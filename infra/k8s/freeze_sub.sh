#!/usr/bin/env bash
# Freeze a submission: fetch from the PVC (md5-verified), validate, make read-only, push to S3.
#   infra/k8s/freeze_sub.sh <SUB-name> <pvc-dir-with-matching_results.tsv-and-candidate_pairs.tsv> [local-dir]
# If [local-dir] is given the files are taken from there instead of the PVC.
# Never overwrites: refuses if submissions/<SUB-name> already exists.
set -euo pipefail
NAME=$1; SRC=$2; LOCAL=${3:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUCKET=${BUCKET:?set BUCKET to the S3 bucket for frozen submissions}
DEST=$ROOT/submissions/$NAME
[ -e "$DEST" ] && { echo "REFUSING: $DEST exists (validated submissions are immutable)"; exit 1; }
mkdir -p "$DEST.partial"
for f in matching_results.tsv candidate_pairs.tsv; do
  if [ -n "$LOCAL" ]; then cp "$LOCAL/$f" "$DEST.partial/$f"; else
    R=$(kubectl exec ber-util --request-timeout=300s -- md5sum "$SRC/$f" | cut -d' ' -f1)
    for try in 1 2 3; do
      kubectl exec ber-util --request-timeout=900s -- cat "$SRC/$f" > "$DEST.partial/$f"
      L=$(md5 -q "$DEST.partial/$f"); [ "$L" = "$R" ] && break
      echo "md5 mismatch on $f (try $try): local $L remote $R"; [ $try = 3 ] && exit 1
    done
  fi
done
cd "$DEST.partial"
python3 $ROOT/resources/student_resource/utils/validate_submission.py --matching matching_results.tsv \
  --candidate candidate_pairs.tsv --test-dir $ROOT/resources/student_resource/dataset/test > VALIDATOR.txt 2>&1 \
  || { echo "VALIDATION FAILED for $NAME"; tail -20 VALIDATOR.txt; exit 1; }
tail -1 VALIDATOR.txt
md5 -r matching_results.tsv candidate_pairs.tsv > MD5SUMS
cd .. && mv "$DEST.partial" "$DEST" && chmod a-w "$DEST"/*
cat "$DEST/MD5SUMS"
AWS_PROFILE=amlc-personal aws s3 cp --recursive --only-show-errors "$DEST" "s3://$BUCKET/ber/final/$NAME/" \
  && echo "pushed s3://$BUCKET/ber/final/$NAME/" || echo "S3 PUSH FAILED for $NAME"
