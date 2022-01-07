import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet18

# 2D CNN encoder using ResNet-18 pretrained
class CNN(nn.Module):
    def __init__(self,args):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(CNN, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.output_size)

    
    def forward(self, x):
        x = self.resnet(x[:,0,...])
        x = x.view(x.size(0),-1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        x = torch.tanh(x)
        
        return x.unsqueeze(1)
    
## ---------------------- end of CRNN module ---------------------- ##