#!/bin/bash
RUNDIR=$HOME/GTX480_rundir
cp -a $HOME/gpgpu-sim_distribution/configs/GTX480/ $RUNDIR
cd $RUNDIR
{ { $@; } > >(tee stdout.txt ); } 2> >( tee stderr.txt >&2 )
echo "GPGPU-Sim finished running \"$@\""
echo "Used rundir=$RUNDIR"
