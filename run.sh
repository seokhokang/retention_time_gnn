#!/bin/sh

python run_pretrain.py

for t in MPI_Symmetry PFR-TK72 FEM_orbitrap_urine FEM_short IPB_Halle FEM_lipids UniToyama_Atlantis FEM_orbitrap_plasma MTBLS87 LIFE_new LIFE_old UFZ_Phenomenex MassBank2 MetaboBase MassBank1 Eawag_XBridgeC18 FEM_long RIKEN Cao_HILIC RIKEN_PlaSMA
do
  for k in {0..9}
  do
    python run_transfer.py -t $t -k $k
  done
done