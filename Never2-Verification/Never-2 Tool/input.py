#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:18:57 2023

@author: surendrakumarreddypolaka
"""

import torch
from pynever.strategies import conversion

# Open pytorch model
dqn_net = conversion.load_network_path('dqn_network.pt')

# Select input to apply
nn_input1 = torch.tensor([-0.112, 1.648])
nn_input2 = torch.tensor([-0.150, 1.990])
nn_input3 = torch.tensor([-0.200, 2.100])
nn_input4 = torch.tensor([-0.100, 1.500])
nn_input5 = torch.tensor([-0.120, 1.700])
nn_input6 = torch.tensor([-0.090, 1.900])
nn_input7 = torch.tensor([ 0.340, 4.300])
nn_input8 = torch.tensor([-0.450, 2.600])


# Noise values 
noise1 = [-0.05]
noise2 = [-0.1]
noise3 = [-0.08]
noise4 = [-0.20]
noise5 = [-0.30]
noise6 = [-0.01]
noise7 = [-0.150]
noise8 = [-1.00]

# Execute the network for the first input
output1 = dqn_net.pytorch_network(nn_input1)
print("Output 1", output1)

# Execute the network for the second input
output2 = dqn_net.pytorch_network(nn_input2)
print("Output 2", output2)

# Execute the network for the third input
output3 = dqn_net.pytorch_network(nn_input3)
print("Output 3", output3)

# Execute the network for the fourth input
output4 = dqn_net.pytorch_network(nn_input4)
print("Output 4", output4)

# Execute the network for the fifth input
output5 = dqn_net.pytorch_network(nn_input5)
print("Output 5", output5)

# Execute the network for the sixth input
output6 = dqn_net.pytorch_network(nn_input6)
print("Output 6", output6)

# Execute the network for the seventh input
output7 = dqn_net.pytorch_network(nn_input7)
print("Output 7", output7)

# Execute the network for the eighth input
output8 = dqn_net.pytorch_network(nn_input8)
print("Output 8", output8)