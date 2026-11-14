#!/usr/bin/env bash
# Robust file transfer between pods / laptop over flaky kubectl websockets: 16 MB chunks, md5 per chunk, retries.
#   infra/k8s/xfer.sh get <pod> <remote-file> <local-file>     pod -> laptop
#   infra/k8s/xfer.sh put <pod> <local-file> <remote-file>     laptop -> pod
# The whole file's md5 is verified at the end; exits non-zero on any mismatch.
set -euo pipefail
MODE=$1; POD=$2; NS=amlc-ber-jy; CH=${CH:-16}
k() { kubectl exec -n $NS "$POD" --request-timeout=300s -- "$@" 2>/dev/null; }
if [ "$MODE" = get ]; then
  SRC=$3; DST=$4
  SIZE=$(k stat -c %s "$SRC"); R=$(k md5sum "$SRC" | cut -d' ' -f1); N=$(( (SIZE + CH*1048576 - 1) / (CH*1048576) ))
  : > "$DST.part"
  for i in $(seq 0 $((N-1))); do
    for t in 1 2 3 4 5; do
      CR=$(k sh -c "dd if='$SRC' bs=1M skip=$((i*CH)) count=$CH 2>/dev/null | md5sum" | cut -d' ' -f1)
      k sh -c "dd if='$SRC' bs=1M skip=$((i*CH)) count=$CH 2>/dev/null" > "$DST.chunk" || true
      [ "$(md5 -q "$DST.chunk")" = "$CR" ] && break
      [ $t = 5 ] && { echo "chunk $i failed"; exit 1; }
    done
    cat "$DST.chunk" >> "$DST.part"
  done
  rm -f "$DST.chunk"; L=$(md5 -q "$DST.part"); [ "$L" = "$R" ] || { echo "MD5 MISMATCH $L $R"; exit 1; }
  mv "$DST.part" "$DST"; echo "get OK $DST $L ($N chunks)"
else
  SRC=$3; DST=$4
  L=$(md5 -q "$SRC"); SIZE=$(stat -f %z "$SRC"); N=$(( (SIZE + CH*1048576 - 1) / (CH*1048576) ))
  k sh -c "rm -f '$DST.part'"
  for i in $(seq 0 $((N-1))); do
    dd if="$SRC" bs=1M skip=$((i*CH)) count=$CH 2>/dev/null > /tmp/xfer.chunk; CL=$(md5 -q /tmp/xfer.chunk)
    for t in 1 2 3 4 5; do
      kubectl exec -i -n $NS "$POD" --request-timeout=300s -- sh -c "cat > '$DST.c$i'" < /tmp/xfer.chunk 2>/dev/null || true
      [ "$(k md5sum "$DST.c$i" | cut -d' ' -f1)" = "$CL" ] && break
      [ $t = 5 ] && { echo "chunk $i failed"; exit 1; }
    done
    k sh -c "cat '$DST.c$i' >> '$DST.part' && rm '$DST.c$i'"
  done
  R=$(k md5sum "$DST.part" | cut -d' ' -f1); [ "$L" = "$R" ] || { echo "MD5 MISMATCH $L $R"; exit 1; }
  k mv "$DST.part" "$DST"; echo "put OK $DST $L ($N chunks)"
fi
