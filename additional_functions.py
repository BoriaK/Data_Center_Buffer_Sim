import numpy as np
import torch
from scipy import signal
from matplotlib import pyplot as plt


def UpsamleZOH(x, up):
    # upsample using ZOH interpolation
    #     up = 5
    spaced_x_up = torch.zeros(up * len(x))
    spaced_x_up[::up] = x
    g_ZOH = torch.ones(up)
    x_up = signal.lfilter(g_ZOH, 1, spaced_x_up)
    return x_up


if __name__ == "__main__":
    rand_arr = torch.rand(10)
    upsampled_rand_arr = torch.from_numpy(UpsamleZOH(rand_arr, 5))
    #
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(rand_arr)
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(upsampled_rand_arr)
    plt.grid()
    plt.show()
