import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path
import torch
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
from additional_functions import UpsamleZOH

# for one stream only:
# without overshoot
Length = 10  # sequence length

alpha = 2  # Threshold slope
B = 60  # total buffer size [packets]
R_max = 100  # the initial Traffic arrival rate [packets/sec]
Traffic = genDataset(d=0.2, seq_len=Length) * R_max  # incoming Traffic [Packets/sec]
# Traffic = torch.ones(Length) * R_max  # incoming Traffic [Packets/sec]  # for debug
# Traffic = torch.Tensor([76.5542, 74.8363, 72.8576, 70.5553, 67.8460])  # for debug
upsample_factor = 21  # up sampling factor for incoming Traffic. [Samples/sec]
Q = torch.zeros(Length * upsample_factor)
Queue_Length_Arr = torch.zeros(Length * upsample_factor)
# Threshold = torch.ones(Length * upsample_factor) * alpha
Threshold = torch.zeros(Length * upsample_factor)
for k in range(Length):
    Delta_Arr = torch.zeros(upsample_factor)
    Rates_Arr = torch.zeros(upsample_factor)
    state = 'transition'  # the transition state at the arrival of a new stream. T(t) > Q(t)
    for t in range(upsample_factor):
        # Q[i] = torch.min(Scaled_Upsampled_Traffic[i], torch.ones(1) * B)
        if state == 'transition':
            Rates_Arr[t] = Traffic[k]
            # Q[t] = Rates_Arr[t] * t / upsample_factor  # [Packets/sample]
            if t == 0:
                Queue_Length_Arr[k * upsample_factor + t] = 0
                Q[k * upsample_factor + t] = 0  # [Packets/sample]
            else:
                Queue_Length_Arr[k * upsample_factor + t] = Queue_Length_Arr[k * upsample_factor + t - 1] + \
                                                            Rates_Arr[t] * 1 / upsample_factor
                Q[k * upsample_factor + t] = Q[k * upsample_factor + t - 1] + Rates_Arr[t] * 1 / upsample_factor  # [Packets/sample]

            Threshold[k * upsample_factor + t] = alpha * (B - Q[k * upsample_factor + t])
            Delta_Arr[t] = Threshold[k * upsample_factor + t] - Queue_Length_Arr[k * upsample_factor + t]

            if Delta_Arr[t] < 0:
                state = 'overshoot'
                Rates_Arr[t] = 0
                # calculate the intersection point between High Priority Threshold and a specific Queue length
                Intr = (Threshold[k * upsample_factor + t - 1] * Queue_Length_Arr[k * upsample_factor + t] -
                        Threshold[k * upsample_factor + t] * Queue_Length_Arr[k * upsample_factor + t - 1]) \
                       / (Threshold[k * upsample_factor + t - 1] - Threshold[k * upsample_factor + t] +
                          Queue_Length_Arr[k * upsample_factor + t] - Queue_Length_Arr[
                              k * upsample_factor + t - 1])
                # Make the correction to Que length, Threshold, and R(t)
                Queue_Length_Arr[k * upsample_factor + t] = Intr
                Threshold[k * upsample_factor + t] = Queue_Length_Arr[k * upsample_factor + t]
                Q[k * upsample_factor + t] = Queue_Length_Arr[k * upsample_factor + t]
                # calculate delta again:
                Delta_Arr[t] = Threshold[k * upsample_factor + t] - Queue_Length_Arr[k * upsample_factor + t]
                if Delta_Arr[t].round(decimals=2) == 0:
                    state = 'steady'
                print('')
        elif state == 'steady':
            Rates_Arr[t] = 0
            Queue_Length_Arr[k * upsample_factor + t] = Queue_Length_Arr[k * upsample_factor + t - 1] + Rates_Arr[
                t] * 1 / upsample_factor  # [Packets/sample]
            Q[k * upsample_factor + t] = Queue_Length_Arr[k * upsample_factor + t]
            Threshold[k * upsample_factor + t] = alpha * (B - Q[k * upsample_factor + t])
            Delta_Arr[t] = Threshold[k * upsample_factor + t] - Queue_Length_Arr[k * upsample_factor + t]

print(Traffic)
# print(Threshold)
# print(Q)

Time = range(0, Length * upsample_factor)
plt.figure()
plt.plot(Time, Q, Time, Threshold)
plt.legend(['q_1', 'T_1'])
plt.grid()
plt.show()
# print(Threshold)


# for multiple streams
