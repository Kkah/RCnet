import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import transforms

from model import gen_model
from opt import parse_args
from utils import *

def CRNN_final_prediction(model, device, loader):

    model.eval()
    
    all_y = []
    all_y_pred1 = []
    
    print("Predicting...")
    with torch.no_grad():
        for image, target in tqdm(loader):
            # distribute data to device
            image =  image.to(device)
            target = target.to(device)
            outputs = model(image)

            all_y.extend(target.cpu().numpy().tolist())
            all_y_pred1.extend(outputs.cpu().numpy().tolist())
            
    return np.array(all_y_pred1), np.array(all_y)

def predict(args):
    # Check GPU to use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")   # use CPU or GPU
    
    # Create model
    model = gen_model(args, device)
    
    # Load model
    print('Loading best model from {}'.format(args.save_model_path))
    model.load_state_dict(torch.load(os.path.join(args.save_model_path,\
            'best_model.pth')))

    # Read and list iamge folders
    image_folder = os.path.join(args.data_path, 'images')
    all_X_list = sorted(os.listdir(image_folder))

    # train, valid split
    train_image_list, valid_image_list, = \
        train_test_split(all_X_list, test_size=0.2, random_state = 0)

    # Mean and standard deviation of training images
    # mean, std = meanstd(data_path, train_image_list)
    mean = [0.567,0.390,0.388]
    std = [0.159,0.155,0.168]

    res_size = 224
    
    # Transform inputs
    transform = A.Compose([A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            transforms.ToTensorV2()])
    
    # Data loading parameters
    params = {'batch_size': 4, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
        if not args.no_cuda else {}

    for exp in valid_image_list:
        
        # read images and targets from a valid experiment
        valid_dataset= read_data([exp], args)
        
        # Dataset
        valid_set = RC_Dataset(valid_dataset, transform, args)

        # DataLoader
        valid_loader = data.DataLoader(valid_set, **params)

        # Prediction
        all_y_pred1, all_y= CRNN_final_prediction(model, device, valid_loader)
        
        # Plot predictions
        fig = plt.figure(figsize=(10, 5))
        plt.title(exp)
        
        plt.subplot(2, 1, 1)
        plt.plot(all_y[...,0])
        plt.plot(all_y_pred1[...,0])
        plt.legend(['target', 'pred'], loc="upper left")
        plt.xlabel('frame')
        plt.ylabel('steer X')

        plt.subplot(2, 1, 2)   
        plt.plot(all_y[...,1])
        plt.plot(all_y_pred1[...,1])
        plt.legend(['target', 'pred'], loc="upper left")
        plt.xlabel('frame')
        plt.ylabel('steer Y')

        title = "./Prediction_{}.png".format(exp)
        save_figure_path = os.path.join(args.save_model_path, 'predictions', 'figures')
        os.makedirs(save_figure_path, exist_ok=True)
        plt.savefig(os.path.join(save_figure_path,title), dpi=600)
        
if __name__=='__main__':

    args = parse_args()
    print('\nPrediction Conditions')
    for arg in vars(args):
        print (arg, ' : ', getattr(args, arg))
    predict(args)
    
    
        