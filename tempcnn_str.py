#
# 
# tempcnn Architektur inspiriert von BreizhCrops
# https://github.com/dl4sits/BreizhCrops/blob/6de796ed36a457c8520322d6110b8f2862fd8c25/breizhcrops/models/TempCNN.py#L58

import torch
import torch.nn as nn

class Conv1D_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=7, drop_probability=0.18203942949809093):
        
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.18203942949809093):
        
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
        )

    def forward(self, X):
        return self.block(X)


class TempCNN(nn.Module):
    def __init__(
        self,
        input_dim=13,
        num_classes=9,
        sequencelength=45,
        kernel_size=7,
        hidden_dims=128,
        dropout=0.18203942949809093,
    ):
        
        super().__init__()
        self.modelname = (
            f"TempCNN_input-dim = {input_dim}_num-classes = {num_classes}_"
            f"sequencelength = {sequencelength}_kernelsize = {kernel_size}_"
            f"hidden-dims={hidden_dims}_dropout = {dropout}")
        self.hidden_dims = hidden_dims
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(
                input_dim, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.flatten = nn.Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(
            hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout
        )
        self.logsoftmax = nn.Sequential(
            nn.Linear(4 * hidden_dims, num_classes),
            nn.LogSoftmax(dim=-1),
        )
        self.T = sequencelength
        self.D = input_dim

    def forward(self, x_ntd):
        x = x_ntd.transpose(1, 2)  
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)        
        x = self.dense(x)          
        return self.logsoftmax(x)  
