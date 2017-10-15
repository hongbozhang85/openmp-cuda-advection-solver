#!/bin/bash

rsync -avz --exclude 'results' * u6170245@sabretooth.anu.edu.au:~
#rsync -avz hz8228@raijin.nci.org.au:~/ass2/openmp/result/* ./openmp/results/
