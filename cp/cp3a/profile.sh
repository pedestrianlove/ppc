#!/usr/bin/env bash
./grading benchmark benchmarks/1.txt
./nsys_wrapper.sh . ./cp benchmarks/3.txt
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i -f rank_0.qdstrm
