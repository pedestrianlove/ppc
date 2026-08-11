#!/bin/bash

set -u

profile_path=$1
shift

RANK="${PMIX_RANK:-0}" 

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
--force-overwrite=true \
-o "$profile_path/rank_$RANK.nsys-rep" \
--trace cuda,nvtx,osrt \
$@
