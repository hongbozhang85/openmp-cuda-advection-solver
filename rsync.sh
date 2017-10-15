#!/bin/bash

rsync -avz --exclude 'results' * hz8228@raijin.nci.org.au:~/ass2/
rsync -avz hz8228@raijin.nci.org.au:~/ass2/openmp/result/* ./openmp/results/
