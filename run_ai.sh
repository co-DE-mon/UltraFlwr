#!/usr/bin/env bash
JOB_NAME=fedavgbccdyli23
runai submit $JOB_NAME \
--backoff-limit 0 \
--image "yang20180924/yli23ultraflwr:latest" \
--gpu 1 \
--project sie-yli23 \
--large-shm \
--cpu 1 \
-v /home/yli23/UltraFlwr:/nfs/home/yang \
--command \
-- bash /nfs/home/yang/scripts/run.sh \
--run-as-user
# --command \
# -- bash /nfs/home/yang/scripts/run.sh \
# --interactive -- sleep infinity \
