#!/bin/bash -x

## this is implemented as a func in .bashrc on cluster
neuro() {
    module purge
    module load python3
    conda activate 2-photon
    which python
    ipython --pylab
}