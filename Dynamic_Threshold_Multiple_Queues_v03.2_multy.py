import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path
import torch
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
from additional_functions import UpsamleZOH

Length = 5  # sequence length

N_ports = 2  # number of ports on the switch
max_Queues = 3  # the maximum number of queues per port
N_streams = [3, 3]  # number of streams on each port <= max_Queues, dim = N_ports
# N_streams = [3]
alpha_high = 2  # alpha for high priority queues
alpha_low = 1  # alpha for low priority queues
B = 60  # total buffer size [packets]
R_max = torch.ones(N_ports, max_Queues) * 60  # the initial Traffic arrival rate [packets/sec]
# R_max = torch.Tensor([[60, 100, 100]])  # for debug
Traffic = torch.zeros(N_ports, max_Queues, Length)
# Generate traffic on each port
for i in range(N_ports):
    for j in range(N_streams[i]):
        # Traffic[i, j, :] = torch.ones(Length) * R_max[i, j]  # generate 1/3 stream on each port
        Traffic[i, j, :] = genDataset(d=0.2, seq_len=Length) * R_max[i, j]  # generate 1/3 stream on each port

upsample_factor = 21  # up sampling factor for incoming Traffic. [Samples/sec]
# num of packets in each queue
Q = torch.zeros(Length * upsample_factor)
Queue_i_Length_Arr = torch.zeros(N_ports, max_Queues, Length * upsample_factor)
Threshold = torch.zeros(2, Length * upsample_factor)  # one threshold for each priority
Diff = torch.zeros(N_ports, max_Queues)
for k in range(Length):  # for each incoming stream in time t
    Rates_Arr = torch.zeros(N_ports, max_Queues, upsample_factor)
    Delta_Arr = torch.zeros(N_ports, max_Queues, upsample_factor)
    # state for each individual queue
    states = [['transition', 'transition', 'transition'], ['transition', 'transition', 'transition']]
    # state for the entire switch
    state = 'transition'
    # the transition state for any priority queue at the arrival of a new stream. T(t) > Q(t)
    # Generate the data rates on each port
    for t in range(upsample_factor):
        Queue_Length_dt = torch.zeros(N_ports, max_Queues)
        # determine R(t) and calculate the Qi(t)):
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if states[i][j] == 'transition':
                    # determine Ri(t), in this model Ri(t) is Ri_in(t) - Ri_out(t)
                    Rates_Arr[i, j, t] = Traffic[i, j, k]
                    # calculate the Qi(t))
                    if t == 0:
                        Queue_Length_dt[i, j] = 0  # each queue is empty at the beginning of arrival of data
                        Queue_i_Length_Arr[
                            i, j, k * upsample_factor + t] = Queue_Length_dt[i, j]
                    else:
                        Queue_Length_dt[i, j] = Rates_Arr[i, j, t] * 1 / upsample_factor  # Qi(dt) = Ri(t) * dt
                        Queue_i_Length_Arr[i, j, k * upsample_factor + t] = Queue_i_Length_Arr[
                                                                                i, j, k * upsample_factor + t - 1] + \
                                                                            Queue_Length_dt[
                                                                                i, j]  # Qi(t) = Qi(t-1) + Qi(dt)

                else:
                    # in steady state Ri_in(t) = Ri_out(t) = Threshold_c(t)
                    Rates_Arr[i, j, t] = 0
                    Queue_Length_dt[i, j] = Rates_Arr[i, j, t] * 1 / upsample_factor  # Qi(dt) = 0
                    Queue_i_Length_Arr[i, j, k * upsample_factor + t] = Queue_i_Length_Arr[
                                                                            i, j, k * upsample_factor + t - 1] + \
                                                                        Queue_Length_dt[i, j]
                    ##############
                    # need to add transient state scenario, whare Ri_in(t) = 0 and Ri_out(t) = C so Ri(t) = -C
                    # Qi(dt) = -C*dt
                    ################
        # calculate the Q(t) and Threshold at each time sample:
        if t == 0:
            Q[k * upsample_factor + t] = 0  # [Packets/sample]
        else:
            Q[k * upsample_factor + t] = Q[k * upsample_factor + t - 1] + Queue_Length_dt.sum((0, 1))

        Threshold[0][k * upsample_factor + t] = alpha_high * (
                B - Q[k * upsample_factor + t])  # set high priority threshold
        Threshold[1][k * upsample_factor + t] = alpha_low * (
                B - Q[k * upsample_factor + t])  # set low priority threshold
        # calculate delta for all streams
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if j == 0:
                    # calculate delta for high priority queue
                    Delta_Arr[i, j, t] = Threshold[0, k * upsample_factor + t] - \
                                         Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                else:
                    # calculate delta for low priority queue
                    Delta_Arr[i, j, t] = Threshold[1, k * upsample_factor + t] - Queue_i_Length_Arr[
                        i, j, k * upsample_factor + t]
        # determine form of correction:
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if Delta_Arr[i, j, t] < 0:
                    Rates_Arr[i, j, t] = 0
                    if Delta_Arr[i, j, t - 1] != 0:  # only if it wasn't in steady state already:
                        if j == 0:  # for high priority queue
                            # calculate the intersection point between High Priority Threshold and a specific Queue length
                            Intr = (Threshold[0, k * upsample_factor + t - 2] * Queue_i_Length_Arr[
                                i, j, k * upsample_factor + t - 1] -
                                    Threshold[0, k * upsample_factor + t - 1] * Queue_i_Length_Arr[
                                        i, j, k * upsample_factor + t - 2]) \
                                   / (Threshold[0, k * upsample_factor + t - 2] - Threshold[
                                0, k * upsample_factor + t - 1] +
                                      Queue_i_Length_Arr[i, j, k * upsample_factor + t - 1] - Queue_i_Length_Arr[
                                          i, j, k * upsample_factor + t - 2])
                        else:  # for low priority queue
                            # calculate the intersection point between Low Priority Threshold and a specific Queue length
                            Intr = (Threshold[1, k * upsample_factor + t - 2] * Queue_i_Length_Arr[
                                i, j, k * upsample_factor + t - 1] -
                                    Threshold[1, k * upsample_factor + t - 1] * Queue_i_Length_Arr[
                                        i, j, k * upsample_factor + t - 2]) \
                                   / (Threshold[1, k * upsample_factor + t - 2] - Threshold[
                                1, k * upsample_factor + t - 1] +
                                      Queue_i_Length_Arr[i, j, k * upsample_factor + t - 1] - Queue_i_Length_Arr[
                                          i, j, k * upsample_factor + t - 2])

                        # Make the correction to Queue length
                        Queue_Length_dt[i, j] = Intr - Queue_i_Length_Arr[i, j, k * upsample_factor + t - 1]
                        Queue_i_Length_Arr[i, j, k * upsample_factor + t] = Intr
                    else:  # it was in steady state already
                        Queue_Length_dt[i, j] = 0
        # re-calculate Q(t) and the Thresholds:
        if t != 0:
            Q[k * upsample_factor + t] = Q[k * upsample_factor + t - 1] + Queue_Length_dt.sum((0, 1))
            Threshold[0, k * upsample_factor + t] = alpha_high * (B - Q[k * upsample_factor + t])
            Threshold[1, k * upsample_factor + t] = alpha_low * (B - Q[k * upsample_factor + t])

        # Update each queue length to be equal to the relevant Threshold
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if Delta_Arr[i, j, t] < 0:
                    # if Delta_Arr[i, j, t - 1] != 0:  # only if it wasn't in steady state already:
                        if j == 0:  # for high priority queue
                            Queue_i_Length_Arr[i, j, k * upsample_factor + t] = Threshold[0, k * upsample_factor + t]
                            # re-calculate delta (just a formality):
                            Delta_Arr[i, j, t] = Threshold[0, k * upsample_factor + t] - \
                                                 Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                        else:
                            # update queue low length:
                            Queue_i_Length_Arr[i, j, k * upsample_factor + t] = Threshold[1][k * upsample_factor + t]
                            # re-calculate low delta:
                            Delta_Arr[i, j, t] = Threshold[1, k * upsample_factor + t] - \
                                                 Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                else:  # only update delta
                    if j == 0:  # for high priority queue
                        Delta_Arr[i, j, t] = Threshold[0, k * upsample_factor + t] - \
                                             Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                    else:
                        Delta_Arr[i, j, t] = Threshold[1, k * upsample_factor + t] - \
                                             Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                # check deltas to determine state:
                if Delta_Arr[i, j, t].round(decimals=2) == 0:
                    states[i][j] = 'steady'
                else:
                    states[i][j] = 'transition'
print(Traffic)

#
# version 2 plot active queues only:
Time = range(0, Length * upsample_factor)
plt.figure()
# fig, axs = plt.subplots(sum(N_streams), 1)
for i in range(N_ports):
    for j in range(N_streams[i]):
        # fig, ax = plt.subplots()
        ax = plt.subplot(sum(N_streams), 1, i * N_streams[i - 1] + j + 1)
        # ax = plt.axes()
        # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(Time, Queue_i_Length_Arr[i, j, :])
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
