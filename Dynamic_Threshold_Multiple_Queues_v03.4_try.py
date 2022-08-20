import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import os.path
import torch
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset

# in this model R(t) is R_in(t) - R_out(t)

Length = 3  # sequence length (time frames)

N_ports = 1  # number of ports on the switch
max_Queues = 2  # the maximum number of queues per port
# N_streams = [3, 2]  # number of streams on each port <= max_Queues, dim = N_ports
N_streams = [2]  # Debug
alpha_high = 2  # alpha for high priority queues
alpha_low = 1  # alpha for low priority queues
B = 60  # total buffer size [packets]
C = 30  # Link Capacity - the rate of packets leaving the buffer
# R_max = torch.ones(N_ports, max_Queues) * 100  # the initial Traffic arrival rate [packets/sec]
R_max = torch.Tensor([[50, 100]])  # for debug 2
# R_max = torch.Tensor([[50, 70]])  # for debug 1
Traffic = torch.zeros(N_ports, max_Queues, Length)
# Generate traffic on each port
for i in range(N_ports):
    for j in range(N_streams[i]):
        # generate stream on each port
        # Traffic[i, j, :] = torch.ones(Length) * R_max[i, j]
        Traffic[i, j, :] = torch.tensor([2, 0, 1]) * R_max[i, j]  # Special case: Q(t)>B
        # Traffic[i, j, :] = genDataset(d=0.2, seq_len=Length) * R_max[i, j]

# Traffic = torch.tensor([[[7.6419, 19.3214, 93.9851, 93.8721, 93.7547]]])

upsample_factor = 11  # up sampling factor for incoming Traffic. [Samples/sec]
# num of packets in each queue
Q = torch.zeros(Length * upsample_factor)
Queue_i_Length_Arr = torch.zeros(N_ports, max_Queues, Length * upsample_factor)
Lost_Packets_dt_Arr = torch.zeros(N_ports, max_Queues, Length * upsample_factor)
Threshold = torch.zeros(2, Length * upsample_factor)  # one threshold for each priority
Diff = torch.zeros(N_ports, max_Queues)
Rates_in_Arr = torch.zeros(N_ports, max_Queues, Length * upsample_factor)
# Delta_Arr = torch.ones(N_ports, max_Queues, Length * upsample_factor) * alpha_high * B  # Initialize Delta to be Maximal
Delta_Arr = torch.zeros(N_ports, max_Queues, Length * upsample_factor)  # Initialize Delta to be zeros
for k in range(Length):  # for each incoming stream in time t
    # Generate the data rates on each port
    for t in range(upsample_factor):
        Queue_Length_dt = torch.zeros(N_ports, max_Queues)
        # determine R(t) and calculate the Qi(t)):
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if k == 0 and t == 0:
                    Rates_in_Arr[i, j, k * upsample_factor + t] = Traffic[i, j, k] - C
                    Queue_Length_dt[i, j] = 0  # each queue is empty at the beginning of arrival of data
                    Queue_i_Length_Arr[
                        i, j, k * upsample_factor + t] = Queue_Length_dt[i, j]

                elif Delta_Arr[i, j, k * upsample_factor + t - 1] > 0 or (k > 0 and Traffic[i, j, k] != Traffic[i, j, k - 1]):
                    # if in transient state
                    Rates_in_Arr[i, j, k * upsample_factor + t] = Traffic[i, j, k] - C
                    # calculate the Qi(t))
                    Queue_Length_dt[i, j] = Rates_in_Arr[
                                        i, j, k * upsample_factor + t] * 1 / upsample_factor  # Qi(dt) = Ri(t) * dt
                    Queue_i_Length_Arr[i, j, k * upsample_factor + t] = torch.max(Queue_i_Length_Arr[
                                        i, j, k * upsample_factor + t - 1] + Queue_Length_dt[i, j], torch.tensor(0))  # Qi(t) = Qi(t-1) + Qi(dt)

                else: # steady state or overshoot
                    # in steady state Ri_in(t) = C = Threshold_c(t)
                    Rates_in_Arr[i, j, k * upsample_factor + t] = 0
                    Queue_Length_dt[i, j] = Rates_in_Arr[i, j, t] * 1 / upsample_factor  # Qi(dt) = 0
                    Queue_i_Length_Arr[i, j, k * upsample_factor + t] = torch.max(Queue_i_Length_Arr[
                                                                                      i, j, k * upsample_factor + t - 1] + \
                                                                                  Queue_Length_dt[i, j],
                                                                                  torch.tensor(0))
                    ##############
                    # need to add transient state scenario, whare Ri_in(t) = 0 and Ri_out(t) = C so Ri(t) = -C
                    # Qi(dt) = -C*dt
                    ################
        # calculate the Q(t) and Threshold at each time sample:
        # if k == 0 and t == 0:
        #     Q[k * upsample_factor + t] = 0  # [Packets/sample]
        # else:
        #     Q[k * upsample_factor + t] = Q[k * upsample_factor + t - 1] + Queue_Length_dt.sum((0, 1))
        Q[k * upsample_factor + t] = Queue_i_Length_Arr[:, :, k * upsample_factor + t].sum(1)
        # Threshold can't be > alpha_c * B
        Threshold[0][k * upsample_factor + t] = torch.min(alpha_high * (
                B - Q[k * upsample_factor + t]), torch.tensor(alpha_high * B))  # set high priority threshold
        Threshold[1][k * upsample_factor + t] = torch.min(alpha_low * (
                B - Q[k * upsample_factor + t]), torch.tensor(alpha_low * B))  # set low priority threshold
        # calculate delta for all streams
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if j == 0:
                    # calculate delta for high priority queue
                    Delta_Arr[i, j, k * upsample_factor + t] = Threshold[0, k * upsample_factor + t] - \
                                                               Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                else:
                    # calculate delta for low priority queue
                    Delta_Arr[i, j, k * upsample_factor + t] = Threshold[1, k * upsample_factor + t] - \
                                                               Queue_i_Length_Arr[
                                                                   i, j, k * upsample_factor + t]
        # determine form of correction:
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if Delta_Arr[i, j, k * upsample_factor + t] < 0:
                    if Delta_Arr[i, j, k * upsample_factor + t - 1] != 0:  # only if it wasn't in steady state already:
                        Rates_in_Arr[i, j, k * upsample_factor + t] = 0  # true for steady state
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
                        ########################################
                        Rates_in_Arr[i, j, k * upsample_factor + t] = Delta_Arr[i, j, k * upsample_factor + t] / (1 / upsample_factor)
                        Lost_Packets_dt_Arr[i, j, k * upsample_factor + t] = -Delta_Arr[i, j, k * upsample_factor + t]
                        if Rates_in_Arr[i, j, k * upsample_factor + t] >= -C:
                            Queue_Length_dt[i, j] = Delta_Arr[i, j, k * upsample_factor + t]  # it doesn't add new packets to total queue
                        else:
                            Queue_Length_dt[i, j] = -C * 1/upsample_factor
                        Queue_i_Length_Arr[i, j, k * upsample_factor + t] = torch.max(Queue_i_Length_Arr[
                                                                                i, j, k * upsample_factor + t - 1] +
                                                                            Queue_Length_dt[i, j],
                                                                            torch.tensor(0))
                            ########################################
        # re-calculate Q(t) and the Thresholds:
        if not (k == 0 and t == 0):
            Q[k * upsample_factor + t] = Queue_i_Length_Arr[:, :, k * upsample_factor + t].sum(1)
            Threshold[0, k * upsample_factor + t] = torch.min(alpha_high * (B - Q[k * upsample_factor + t]),
                                                              torch.tensor(alpha_high * B))
            Threshold[1, k * upsample_factor + t] = torch.min(alpha_low * (B - Q[k * upsample_factor + t]),
                                                              torch.tensor(alpha_low * B))

        # Update each queue length to be equal to the relevant Threshold
        ########need to update according to re-calculated queue_dt
        for i in range(N_ports):
            for j in range(N_streams[i]):
                if Delta_Arr[i, j, k * upsample_factor + t] < 0:
                    # if Delta_Arr[i, j, t - 1] != 0:  # only if it wasn't in steady state already:
                    if j == 0:  # for high priority queue
                        Queue_i_Length_Arr[i, j, k * upsample_factor + t] = Threshold[0, k * upsample_factor + t]
                        # re-calculate delta (just a formality):
                        Delta_Arr[i, j, k * upsample_factor + t] = Threshold[0, k * upsample_factor + t] - \
                                                                   Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                    else:
                        # update queue low length:
                        Queue_i_Length_Arr[i, j, k * upsample_factor + t] = Threshold[1][k * upsample_factor + t]
                        # re-calculate low delta:
                        Delta_Arr[i, j, k * upsample_factor + t] = Threshold[1, k * upsample_factor + t] - \
                                                                   Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                else:  # only update delta
                    if j == 0:  # for high priority queue
                        Delta_Arr[i, j, k * upsample_factor + t] = Threshold[0, k * upsample_factor + t] - \
                                                                   Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                    else:
                        Delta_Arr[i, j, k * upsample_factor + t] = Threshold[1, k * upsample_factor + t] - \
                                                                   Queue_i_Length_Arr[i, j, k * upsample_factor + t]
                # check deltas to determine state:
                # if Delta_Arr[i, j, k * upsample_factor + t].round(decimals=2) == 0:
                #     states[i][j] = 'steady'
                # else:
                #     states[i][j] = 'transition'
print(Traffic)
print(Lost_Packets_dt_Arr)

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
        plt.plot(Time, Queue_i_Length_Arr[i, j, :], 'g--')
        if j == 0:
            plt.plot(Time, Threshold[0][:], '+r:')
            plt.legend(['q_' + str(i) + '_h', 'T_' + str(i) + '_h'])
        else:
            plt.plot(Time, Threshold[1][:], '+b:')
            plt.legend(['q_' + str(i) + '_l_' + str(j), 'T_' + str(i) + '_l'])
        plt.xlabel('Time [samples]')
        plt.ylabel('Packets')
        plt.grid()
plt.show()
