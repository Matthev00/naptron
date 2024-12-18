#! /bin/bash

# Clearn data
# rm -rf data/

# # Download COCO dataset
# python scripts/download_coco_train.py
# python scripts/download_coco_ood.py

# First NAPTRON step
./1_naptron.sh

# Second NAPTRON step
./2_naptron.sh

# Third NAPTRON step
./3_naptron.sh
