#!/usr/bin/env bash
./grading benchmark benchmarks/1.txt
./nsys_wrapper.sh . ./cp benchmarks/3.txt
rm rank_0.nsys-rep
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i rank_0.qdstrm
nsys stats --report nvtxsum rank_0.nsys-rep --timeunit=s
