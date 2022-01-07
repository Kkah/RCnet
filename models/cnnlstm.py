import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet18

class EncoderCNN(nn.Module):
    def __init__(self,args):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :]) 
            x = x.view(x.size(0), -1) 
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=1)
        
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, args):
        super(DecoderRNN, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=args.CNN_embed_dim,
            hidden_size=args.h_RNN,
            num_layers=args.num_layers,
            batch_first = True
        )

        self.fc1 = nn.Linear(args.h_RNN, args.h_FC_dim)
        self.fc2 = nn.Linear(args.h_FC_dim, args.output_size)

    def forward(self, x):
        self.LSTM.flatten_parameters()
        out, (h_n, h_c) = self.LSTM(x)
        x = self.fc1(out[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x.unsqueeze(1)
    
class CNNLSTM(nn.Module):
    def __init__(self, args):
        super(CNNLSTM, self).__init__()
        self.cnn_encoder = EncoderCNN(args)
        self.rnn_decoder= DecoderRNN(args)
        
    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.rnn_decoder(x)
        return x