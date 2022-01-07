import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import *

import albumentations as A
from albumentations.pytorch import transforms
from models import *


if __name__=='__main__':
        # set path
    data_path = "/Users/217/RC/dataset"    # define UCF-101 RGB data path
    save_model_path = "./model"

    # CNN_Encoder architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 256, 128
    CNN_features = 2   # latent dim extracted by 2D CNN
    res_size = 224        # ResNet image size
    dropout_p = 0.5     # dropout probability

    # training parameters
    batch_size = 256
    seq_len = 1
    target_len = 1


    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    
    # Create model
    model = CNN_Encoder(
        fc_hidden1=CNN_fc_hidden1,
        fc_hidden2=CNN_fc_hidden2, 
        drop_p=dropout_p, 
        CNN_embed_dim=CNN_features
        ).to(device)
                    
    image_folder = os.path.join(data_path, 'images')
    history_folder = os.path.join(data_path,'history')

    print("Extracting lists of dataset...\n")

    all_X_list = sorted(os.listdir(image_folder))
    all_y_list = sorted(os.listdir(history_folder))

    print("Experiment List : " , all_X_list, "\n")

    print("Split the dataset...\n")

    # train, valid split
    train_image_list, valid_image_list, train_history_list, valid_history_list = \
        train_test_split(all_X_list, all_y_list, test_size=0.2, random_state = 0)

    model.load_state_dict(torch.load(os.path.join(save_model_path,\
        'best_model.pth')))
        
    model.eval()
        
    print(valid_image_list)
    
    # mean, std = meanstd(data_path, train_image_list)
    mean = [0.564,0.386,0.377]
    std = [0.173,0.166,0.179]
    
    transform = A.Compose([A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            transforms.ToTensorV2()])
    
    for fname in valid_image_list:
        plt.ion()
        fig, ax = plt.subplots(3,1)
        cap = cv2.VideoCapture('{}.avi'.format(fname))
               
        pred = []
        i = 0
        
        history = pd.read_csv(os.path.join(data_path,'history','{}.csv'.format(fname)))
        targets = history[['.joy_x','.joy_y']][1:].values

        num_frame = len(targets)
        
        image = np.zeros((720,1280,3))
        img = ax[0].imshow(image)
        linetx, = ax[1].plot([],[],'tab:blue')
        linepx, = ax[1].plot([],[],'tab:red')
        linety, = ax[2].plot([],[],'tab:blue')
        linepy, = ax[2].plot([],[],'tab:red')
         
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret:   
                h,w,c = image.shape
                image_large = cv2.resize(image, (720,1280))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = transform(image = image_rgb)['image'].unsqueeze(0).to(device)
                
                with torch.no_grad(): 
                    output = model(image_tensor).cpu()
                    
                x = -output[...,0]
                y = -output[...,1]
                
                x = int(x*w/2 + w/2)
                y = int(y*h/2 + h/2)
                
                pred.append(output)
                out = torch.stack(pred,1)

                linetx.set_data(np.arange(i+1), targets[:i+1,0])
                linepx.set_data(np.arange(i+1), out[...,0])
                
                linety.set_data(np.arange(i+1), targets[:i+1,1])
                linepy.set_data(np.arange(i+1), out[...,1])
                
                i += 1
                ax[0].imshow(image_rgb)
                ax[0].set_axis_off()
                ax[1].set_ylim(-1,1)    
                ax[1].set_xlim(-10,num_frame+10)
                ax[1].legend(['target','predict'])
                ax[2].set_ylim(-1,1)
                ax[2].set_xlim(-10,num_frame+10)
                ax[2].legend(['target','predict'])
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            else:
                break
                
        cap.release()
        cv2.destroyAllWindows()

        
    

