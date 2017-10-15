#!/bin/bash

env LD_LIBRARY_PATH="/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/src" ./bin/Release/clasificador -e "data/X.dat" -c "data/yMapped.mat" -d "data/"
cd data
octave analizador.m
rm costo.mat
cd ..

exit 0
