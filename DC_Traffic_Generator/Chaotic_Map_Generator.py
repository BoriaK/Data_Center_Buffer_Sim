import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path
import torch


def genDataset(d, seq_len):
    attempts = 0
    ZC = 0
    while ZC < 0.1 * seq_len and attempts < 10:
        # 1.5 < m1,m2 <= 2
        m1 = 2
        m2 = 2
        # 0 < d < 1, small d -> mostly mice, large d -> mostly elephants
        # d = 0.5

        # randomly generate initial network state x_0 from uniform distribution: x_0~U(0,1)
        x0 = torch.rand(1) * 0.99 + 0.01
        x = torch.zeros(seq_len + 1, dtype=torch.float)
        x[0] = x0

        for i in range(0, seq_len):
            if x[i] <= d:
                x[i + 1] = x[i] + (1 - d) * np.power((x[i] / d), m1)
            else:
                x[i + 1] = x[i] - d * np.power((1 - x[i]) / (1 - d), m2)
        x = x[1:]  # unscaled data [0,1)
        # x_upscaled = x * 20
        # Check the number of zero crossings, to avoid "flat" data set generation
        ##########################################
        x_norm = x - 0.5
        # Get the total number of zero crossings of the signal
        ZC = (np.diff(np.sign(x_norm.data)) != 0).sum()

        attempts += 1
        if ZC < 0.1 * seq_len and attempts == 10:
            raise ValueError("problem with data generation")

    return x


if __name__ == "__main__":
    x = genDataset(0.2, 1000)
    import matplotlib.pyplot as plt

    plt.plot(x.view(-1))
    plt.show()
