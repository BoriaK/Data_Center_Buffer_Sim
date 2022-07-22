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
N_streams = [1, 3]
Length = 1000  # sequence length
alpha_high = 2  # alpha for high priority queues
alpha_low = 1  # alpha for low priority queues
B = 60  # full buffer capacity (for a single switch), 60 packets
Traffic = torch.zeros(N_ports, max_Queues, Length)
# print(Q.shape)
for i in range(N_ports):
    for j in range(N_streams[i]):
        Traffic[i, j, :] = genDataset(d=0.2, seq_len=1000)  # generate 1/3 stream on port1
        # Traffic[1, 0, :] = genDataset(d=0.2, seq_len=1000)  # generate 3 different streams on port2
        # Traffic[1, 1, :] = genDataset(d=0.2, seq_len=1000)
        # Traffic[1, 2, :] = genDataset(d=0.2, seq_len=1000)

Q = torch.min(Traffic.sum((0, 1)), torch.ones(Length) * 60)

Threshold_h = alpha_high * (B - Q)
Threshold_l = alpha_low * (B - Q)
# Threshold_1h = alpha_high * (B - Q)
# Threshold_2h = alpha_high * (B - Q)
# Threshold_2l = alpha_low * (B - Q)

Time = range(0, len(Traffic[0, 0, :]))
plt.figure()
for k in range(N_ports * max_Queues):
    plt.subplot(N_ports * max_Queues, 1, k)
    if k == 0 or k == 3:
        plt.plot(Time, Traffic[0, 0, :], Time, Threshold_h)
        plt.legend(['q_1_h', 'T_1_h'])
    plt.xlabel('Time [samples]')
    plt.ylabel('Packets')
    plt.grid()

    # plt.subplot(4, 1, 2)
    # plt.plot(Time, Traffic[1, 0, :], Time, Threshold_2h)
    # plt.xlabel('Time [samples]')
    # plt.ylabel('Packets')
    # plt.grid()
    # plt.legend(['q_2_h', 'T_2_h'])
    # plt.subplot(4, 1, 3)
    # plt.plot(Time, Traffic[1, 1, :], Time, Threshold_2l)
    # plt.xlabel('Time [samples]')
    # plt.ylabel('Packets')
    # plt.grid()
    # plt.legend(['q_2_l_1', 'T_2_l'])
    # plt.subplot(4, 1, 4)
    # plt.plot(Time, Traffic[1, 2, :], Time, Threshold_2l)
    # plt.xlabel('Time [samples]')
    # plt.ylabel('Packets')
    # plt.grid()
    # plt.legend(['q_2_l_2', 'T_2_l'])
    plt.show()

# print(Threshold.shape)

# check if threshold(alpha) is optimal, sum over all Q's in time t needs to be equal 1 -> Max Throughput
# TpT = Threshold.sum(0)

# print(TpT)
# plt.plot(TpT.view(-1))
# plt.show()

# print('')
