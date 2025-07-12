# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:03:17 2022

@author: xredin00
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

# definice sítě
class MLP_net(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size):
        super(MLP_net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_2_size, hidden_3_size)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_3_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    

def MyModel(data):

#     Funkce slouzi k implementaci nauceneho modelu. Vas model bude ulozen v samostatne promenne a se spustenim se aplikuje
#     na vstupni data. Tedy, model se nebude pri kazdem spousteni znovu ucit. Ostatni kod, kterym doslo k nauceni modelu,
#     take odevzdejte v ramci projektu.

#Vstup:             data:           vstupni data reprezentujici 1
#                                   objekt (1 pacienta, 1 obrazek, apod.). 

#Vystup:            output:         zarazeni objektu do tridy

    # nastaveni zarizeni
    device = "cpu"

    # architektura siete
    input_size = 28 * 28
    hidden_1_size = 512
    hidden_2_size = 256
    hidden_3_size = 128
    output_size = 10
    mymodel = MLP_net(input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size).to(device)

    # nahrávání váh (modelu)
    mymodel.load_state_dict(torch.load('mymodel_3.pth', map_location=device, weights_only=True))

    #urceni triedy obrazka
    mymodel.eval()
    with torch.no_grad():
        data = data.clone().detach().to(device)
        data = data.view(-1, input_size)
        output = mymodel(data)

        #spustenie predikovania
        _, predicted_class = torch.max(output, 1)

    # vystup trieda ako integer
    output = predicted_class.item()
    
    return output

    
    