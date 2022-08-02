import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path
import torch
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
from additional_functions import UpsamleZOH

Length = 1  # sequence length

N_ports = 2  # number of ports on the switch
max_Queues = 3  # the maximum number of queues per port
N_streams = [3, 2]  # number of streams on each port <= max_Queues, dim = N_ports
alpha_high = 2  # alpha for high priority queues
alpha_low = 1  # alpha for low priority queues
B = 60  # total buffer size [packets]
R_max = torch.ones(N_ports, max_Queues) * 50  # the initial Traffic arrival rate [packets/sec]
Traffic = torch.zeros(N_ports, max_Queues, Length)
# Traffic = genDataset(d=0.2, seq_len=Length) * R_max  # incoming Traffic [Packets/sec]
# Traffic = torch.Tensor([76.5542, 74.8363, 72.8576, 70.5553, 67.8460])  # for debug
# Generate traffic on each port
for i in range(N_ports):
    for j in range(N_streams[i]):
        Traffic[i, j, :] = torch.ones(Length) * R_max[i, j]  # generate 1/3 stream on each port
        # Traffic[i, j, :] = genDataset(d=0.2, seq_len=Length) * R_max[i, j]  # generate 1/3 stream on each port

upsample_factor = 11  # up sampling factor for incoming Traffic. [Samples/sec]
Q = torch.zeros(Length * upsample_factor)
Threshold = torch.zeros(2, Length * upsample_factor)  # one threshold for each priority
Diff = torch.zeros(N_ports, max_Queues)
for k in range(Length):  # for each incoming stream in time t
    Delta_Arr = torch.zeros(N_ports, max_Queues, upsample_factor)
    Rates_Arr = torch.zeros(N_ports, max_Queues, upsample_factor)
    state = [['transition', 'transition', 'transition'], ['transition', 'transition', 'transition']]
    # state_high = 'transition'  # the transition state for high priority queue at the arrival of a new stream. T(t) > Q(t)
    # state_low = 'transition'  # the transition state for low priority queue at the arrival of a new stream. T(t) > Q(t)
    # Generate the data rates on each port
    for t in range(upsample_factor):
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if state[i][j] == 'transition':
                    # Q[i] = torch.min(Scaled_Upsampled_Traffic[i], torch.ones(1) * B)
                    Rates_Arr[i, j, t] = Traffic[i, j, k]
                elif state[i][j] == 'overshoot':
                    if j == 0:
                        alpha = alpha_high
                    else:
                        alpha = alpha_low
                    t1 = (alpha * B - (1 + alpha) * Q[k * upsample_factor + t - 2]) / (
                                (1 + alpha) * torch.abs(Rates_Arr[i, j, t - 1]))
                    t2 = ((1 / upsample_factor) - t1)  # the relative time it took to pass the equalibrium
                    # t2 = torch.abs(1 - t1)
                    x1 = torch.sqrt(Rates_Arr[i, j, t - 1] ** 2 - 1) * t2
                    Rates_Arr[i, j, t] = torch.sign(Delta_Arr[i, j, t - 1]) * x1 * upsample_factor  # rate of correction
                    #######################################################################
        # calculate the Q(t))
        if t == 0:
            Q[k * upsample_factor + t] = 0  # [Packets/sample]
        else:
            Q[k * upsample_factor + t] = Q[k * upsample_factor + t - 1] + Rates_Arr.sum((0, 1))[
                t] * 1 / upsample_factor  # [Packets/sample]

        Threshold[0][k * upsample_factor + t] = alpha_high * (
                B - Q[k * upsample_factor + t])  # set high priority threshold
        Threshold[1][k * upsample_factor + t] = alpha_low * (
                B - Q[k * upsample_factor + t])  # set low priority threshold
        # calculate delta
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if j == 0:
                    # calculate delta for high priority queue
                    Delta_Arr[i, j, t] = Threshold[0, k * upsample_factor + t] - Q[k * upsample_factor + t]
                else:
                    # calculate delta for low priority queue
                    Delta_Arr[i, j, t] = Threshold[1, k * upsample_factor + t] - Q[k * upsample_factor + t]
                if Delta_Arr[i, j, t] <= 0:
                    state[i][j] = 'overshoot'
                print('')
    # elif state_high == 'transition' and state_low == 'overshoot':
    #     for i in range(N_ports):
    #         for j in range(N_streams[i]):
    #             if j == 0:  # for the high priority queues, which are still not full
    #                 Rates_Arr[i, j, t] = Traffic[i, j, k]
    #             else:
    #                 # add per port - per queue calculation:
    #                 t1 = (alpha_low * B - (1 + alpha_low) * Q[k * upsample_factor + t - 2]) / (
    #                             (1 + alpha_low) * torch.abs(Rates_Arr[t - 1])) # the relative time it took to get to equalibrium
#                     t2 = ((1 / upsample_factor) - t1)  # the relative time it took to pass the equalibrium
#                     # t2 = torch.abs(1 - t1)
#                     x1 = torch.sqrt(Rates_Arr[t - 1] ** 2 - 1) * t2
#                     Rates_Arr[t] = torch.sign(Delta_Arr[t - 1]) * x1 * upsample_factor  # rate of correction
#                     Q[k * upsample_factor + t] = Q[k * upsample_factor + t - 1] + Rates_Arr[
#                         t] * 1 / upsample_factor  # [Packets/sample]
#                     # Q[t] = Q[t].round(decimals=2)
#                     Threshold[k * upsample_factor + t] = alpha * (B - Q[k * upsample_factor + t])
#                     Delta_Arr[t] = Threshold[k * upsample_factor + t] - Q[k * upsample_factor + t]
#                     if Delta_Arr[t].round(decimals=2) == 0:
#                         state = 'steady'
#                 elif state == 'steady':
#                     Rates_Arr[t] = 0
#                     Q[k * upsample_factor + t] = Q[k * upsample_factor + t - 1] + Rates_Arr[
#                         t] * 1 / upsample_factor  # [Packets/sample]
#                     Threshold[k * upsample_factor + t] = alpha * (B - Q[k * upsample_factor + t])
#                     Delta_Arr[t] = Threshold[k * upsample_factor + t] - Q[k * upsample_factor + t]
#
# print(Traffic)
# # print(Threshold)
# # print(Q)
#
# Time = range(0, Length * upsample_factor)
# plt.figure()
# plt.plot(Time, Q, Time, Threshold)
# plt.legend(['q_1', 'T_1'])
# plt.grid()
# plt.show()
# # print(Threshold)
#
#
# # for multiple streams
