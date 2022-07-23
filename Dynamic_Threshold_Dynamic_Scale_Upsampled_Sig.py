import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path
import torch
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
from additional_functions import UpsamleZOH

# for one stream only:
# Length = 100  # sequence length
# alpha = 1
# B = 60
# Scale_init = 60
# Traffic = genDataset(d=0.2, seq_len=Length)
# Scaled_Traffic = torch.zeros(Length)
# Q = torch.zeros(Length)
# Threshold = torch.zeros(Length)
# Scale_Array = [Scale_init]
# for i in range(Length):
#     Scaled_Traffic[i] = Traffic[i] * Scale_Array[i]
#     Q[i] = torch.min(Scaled_Traffic[i], torch.ones(1) * 60)
#     Threshold[i] = alpha * (B - Q[i])
#     Diff = Threshold[i] - Scaled_Traffic[i]
#     # if Diff < 0:
#     #     New_Scale = 0
#     # else:
#     #     New_Scale = Scale_Array[i] + Diff
#     New_Scale = Scale_Array[i] + Diff
#     Scale_Array.append(New_Scale)

# print('')

# Time = range(0, len(Traffic))
# plt.figure()
# plt.plot(Time, Scaled_Traffic, Time, Threshold)
# plt.legend(['q_1', 'T_1'])
# plt.grid()
# plt.show()
# print(Threshold)

# for multiple streams

N_ports = 2  # number of ports on the switch
max_Queues = 3  # the maximum number of queues per port
# N_streams = [3, 3]  # number of streams on each port <= max_Queues, dim = N_ports
N_streams = [3, 2]  # number of streams on each port <= max_Queues, dim = N_ports
Length = 5  # sequence length
alpha_high = 2  # alpha for high priority queues
alpha_low = 1  # alpha for low priority queues
B = 60  # full buffer capacity (for a single switch), 60 packets
Scale_init = torch.ones(N_ports, max_Queues) * 20
Traffic = torch.zeros(N_ports, max_Queues, Length)
upsample_factor = 5
Upsampled_Traffic = torch.zeros(N_ports, max_Queues, Length * upsample_factor)
Scaled_Upsampled_Traffic = torch.zeros(N_ports, max_Queues, Length * upsample_factor)
Q = torch.zeros(Length * upsample_factor)
Threshold = torch.zeros(2, Length * upsample_factor)
Diff = torch.zeros(N_ports, max_Queues)
Scale_Array = [Scale_init]
# Generate unscaled traffic on each port
for i in range(N_ports):
    for j in range(N_streams[i]):
        Traffic[i, j, :] = genDataset(d=0.2, seq_len=Length)  # generate 1/3 stream on each port
        Upsampled_Traffic[i, j, :] = torch.from_numpy(UpsamleZOH(Traffic[i, j, :], upsample_factor))

for k in range(Length * upsample_factor):
    # first pass over all the ports and queues - calculate the threshold
    for i in range(N_ports):
        for j in range(N_streams[i]):
            Scaled_Upsampled_Traffic[i, j, k] = Upsampled_Traffic[i, j, k] * Scale_Array[k][i][j]
    Q[k] = torch.min(Scaled_Upsampled_Traffic.sum((0, 1))[k], torch.ones(1) * 60)
    Threshold[0][k] = alpha_high * (B - Q[k])  # set high priority threshold
    Threshold[1][k] = alpha_low * (B - Q[k])  # set low priority threshold
    # second pass over all the ports and queues - calculate the adjustment to scale
    for i in range(N_ports):
        for j in range(N_streams[i]):
            if j == 0:
                Diff[i, j] = Threshold[0][k] - Scaled_Upsampled_Traffic[i, j, k]
            else:
                Diff[i, j] = Threshold[1][k] - Scaled_Upsampled_Traffic[i, j, k]
            # if Diff < 0:
            #     New_Scale = 0
            # else:
            #     New_Scale = Scale_Array[i] + Diff
    New_Scale = Scale_Array[k] + Diff
    Scale_Array.append(New_Scale)

Time = range(0, Length * upsample_factor)
# version 1: plot all ports
# plt.figure()
# for i in range(N_ports):
#     for j in range(max_Queues):
#         plt.subplot(N_ports * max_Queues, 1, i * max_Queues + j + 1)
#         plt.plot(Time, Scaled_Traffic[i, j, :])
#         if j == 0:
#             plt.plot(Time, Threshold[0][:])
#             plt.legend(['q_' + str(i) + '_h', 'T_' + str(i) + '_h'])
#         else:
#             plt.plot(Time, Threshold[1][:])
#             plt.legend(['q_' + str(i) + '_l_' + str(j), 'T_' + str(i) + '_l'])
#         plt.xlabel('Time [samples]')
#         plt.ylabel('Packets')
#         plt.grid()
# plt.show()

# version 2 plot active queues only:
plt.figure()
for i in range(N_ports):
    for j in range(N_streams[i]):
        plt.subplot(sum(N_streams), 1, i * N_streams[i - 1] + j + 1)
        plt.plot(Time, Scaled_Upsampled_Traffic[i, j, :])
        if j == 0:
            plt.plot(Time, Threshold[0][:])
            plt.legend(['q_' + str(i) + '_h', 'T_' + str(i) + '_h'])
        else:
            plt.plot(Time, Threshold[1][:])
            plt.legend(['q_' + str(i) + '_l_' + str(j), 'T_' + str(i) + '_l'])
        plt.xlabel('Time [samples]')
        plt.ylabel('Packets')
        plt.grid()
plt.show()

# print(Threshold.shape)


# print('')
