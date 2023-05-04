#!/bin/bash
rm  run.abo*
rm *.nc
# OpenMp Environment
export OMP_NUM_THREADS=1
# Commands before execution
export OMP_NUM_THREADS=1
#export PATH=$HOME/git_repos/abinit/_build/src/98_main:$PATH

#mpirun  -n 1 /home/hexu/projects/abinit_wann/build/src/98_main/anaddb < /home/hexu/projects/abiwanntest/BTO_wann/wann/run.files > /home/hexu/projects/abiwanntest/BTO_wann/wann/run.log 2> /home/hexu/projects/abiwanntest/BTO_wann/wann/run.err
mpirun  -n 1 /home/hexu/projects/abinit/build/src/98_main/anaddb < run.files # > run.log 2>run.err

#mpirun  -n 1 /home/hexu/projects/abinit/build/src/98_main/anaddb < /home/hexu/projects/abiwanntest/BTO_wann/wann/run.files 
