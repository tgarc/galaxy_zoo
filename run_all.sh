#!/bin/bash
################################################################################
# Dylan Anderson
# University of Texas at Austin
# May 6, 2014
# 
# This script is a quick and dirty way to launch the image processing code
# in parallel over the training set.
################################################################################


python code/image_features.py -i Data/images_training_rev1/ -n 15395 -o 15395 -s results/out2.csv &
python code/image_features.py -i Data/images_training_rev1/ -n 15395 -o 30790 -s results/out3.csv &
python code/image_features.py -i Data/images_training_rev1/ -n 15394 -o 46184 -s results/out4.csv &
