import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path
import torch
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset

# for one stream only:
alpha = 1
B = 1
Q = genDataset(d=0.2, seq_len=1000)
Threshold = alpha * (B - Q)

print(Threshold)
