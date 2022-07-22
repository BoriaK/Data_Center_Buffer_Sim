import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path
import torch
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset

# # for one stream only:
# alpha = 1
# B = 1
# Q = genDataset(d=0.2, seq_len=1000)
# Threshold = alpha * (B - Q)
#
# print(Threshold)

# for multiple streams

N_ports = 2  # number of ports on the switch
max_Queues = 3  # the maximum number of queues per port
# N_streams = torch.randint(max_Queues, (N_ports,))  # number of streams on each port
N_streams = [3, 3]  # number of streams on each port
Length = 100  # sequence length
alpha_high = 2  # alpha for high priority queues
alpha_low = 1  # alpha for low priority queues
B = 60  # full buffer capacity (for a single switch), 60 packets
Scale_init = 20
Traffic = torch.zeros(N_ports, max_Queues, Length)
# print(Q.shape)
for i in range(N_ports):
    for j in range(N_streams[i]):
        Traffic[i, j, :] = genDataset(d=0.2, seq_len=Length) * Scale_init  # generate 1/3 stream on each port
Q = torch.min(Traffic.sum((0, 1)), torch.ones(Length) * 60)

Threshold_h = alpha_high * (B - Q)
Threshold_l = alpha_low * (B - Q)
# Threshold_1h = alpha_high * (B - Q)
# Threshold_2h = alpha_high * (B - Q)
# Threshold_2l = alpha_low * (B - Q)

Time = range(0, len(Traffic[0, 0, :]))
plt.figure()
for i in range(N_ports):
    for j in range(max_Queues):
        plt.subplot(N_ports * max_Queues, 1, i * max_Queues + j + 1)
        plt.plot(Time, Traffic[i, j, :])
        if j == 0:
            plt.plot(Time, Threshold_h)
            plt.legend(['q_' + str(i) + '_h', 'T_' + str(i) + '_h'])
        else:
            plt.plot(Time, Threshold_l)
            plt.legend(['q_' + str(i) + '_l_' + str(j), 'T_' + str(i) + '_l'])
        plt.xlabel('Time [samples]')
        plt.ylabel('Packets')
        plt.grid()
plt.show()

# print(Threshold.shape)

# check if threshold(alpha) is optimal, sum over all Q's in time t needs to be equal 1 -> Max Throughput
# TpT = Threshold.sum(0)

# print(TpT)
# plt.plot(TpT.view(-1))
# plt.show()

# print('')
