#!/bin/bash
#KRS TEST DRIVER

SIZE="1024 2048 32768 524288 8388608"
TEAMS="1 2 4 8 32 64 128"
TS="1 2 4 8 16 32 64"

for T in $TEAMS;do
  for TS in $TS;do 
    for S in $SIZE;do
     echo "======$T,$TS,$S======="
      mpirun -np 1 -x CUDA_VISIBLE_DEVICES=0 -npernode 1 ./test --kokkos-map-device-id-by=mpi_rank : -np 1 -x CUDA_VISIBLE_DEVICES=1 -npernode 1 ./test --kokkos-map-device-id-by=mpi_rank
    done
  done
done
