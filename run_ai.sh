#!/usr/bin/env bash
JOB_NAME=test17bccd3clientsyli23
runai submit $JOB_NAME \
--backoff-limit 0 \
--image "yang20180924/yli23ultraflwr:latest" \
--gpu 1 \
--project sie-yli23 \
--large-shm \
--cpu 1 \
-v /home/yli23/UltraFlwr:/nfs/home/yang \
--command \
-- bash /nfs/home/yang/FedYOLO/test/test.sh \
--run-as-user
# --command \
# -- bash /nfs/home/yang/FedYOLO/test/test.sh \
# --command \
# -- bash /nfs/home/yang/scripts/benchmark.sh \
# --interactive -- sleep infinity \
