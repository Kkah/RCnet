from functools import total_ordering
import os
import numpy as np
import torch
from torch.utils import data
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import transforms
from RCDataset import RC_Dataset
## ---------------------- end of Dataloaders ---------------------- ##

def read_data(X_list, args):
    
    image_path = args.data_path + '/images'
    history_path = args.data_path + '/history'
    
    # Targets to perdict for history
    targets = ['.joy_x', '.joy_y']
    
    # Number of frames to skip
    start = args.start_skip_frame
    end = -args.end_skip_frame
    
    image_seq = []
    target_seq = []

    for exp in X_list:
        
        # Get all directory of frames
        frame_list = sorted(os.listdir(os.path.join(image_path, exp)))            
        frame_path = [os.path.join(image_path,exp,frame) for frame in frame_list[start : end]]
        
        # check number of sequences can be made
        num_seq = len(frame_path) - args.seq_len - args.target_len + 2
        
        # make sequences of images to read
        image_seq += [frame_path[i : i + args.seq_len] for i in range(num_seq)]
        
        # read history csv file
        history = pd.read_csv(os.path.join(history_path, exp) + '.csv')
        
        # make history = frame + 1
        while len(history) != len(frame_list) + 1  :
            history = history[:-1]
        
        # difference of position
        df = history[targets].diff()[start : end]
        
        # make sequences of target variables
        target_seq += [torch.FloatTensor(df[i + args.seq_len - 1 : i + args.seq_len + args.target_len - 1].values) for i in range(num_seq)]
    
    # column names of images
    img_columns = ['image{}'.format(i) for i in range(args.seq_len)]
    
    # make dataframe from sequences
    image = pd.DataFrame(data = image_seq)
    image.columns = img_columns
    
    target = pd.DataFrame(data = target_seq)
    target.columns = ['target']
    
    # concatnate image and target to make dataset
    dataset = pd.concat([image, target], axis = 1)
    dataset.index = np.arange(0, len(dataset))
    
    return dataset

def load_data(args):
    
    # path to images
    image_folder = os.path.join(args.data_path, 'images')

    # read all image folders and sort
    all_X_list = sorted(os.listdir(image_folder))

    # train, valid split
    train_image_list, valid_image_list = \
        train_test_split(all_X_list, test_size=0.2, random_state = 0)
    
    # make 
    train_dataset = read_data(train_image_list, args)
    valid_dataset = read_data(valid_image_list, args)
    
    # mean and standard deviation to normalize images
    # mean, std = mean_std(args, train_image_list)
    mean = [0.567,0.390,0.388]
    std = [0.159,0.155,0.168]

    # size of inputs(images)
    res_size = 224
    
    train_transform = A.Compose([ A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                            transforms.ToTensorV2()])
    
    valid_transform = A.Compose([ A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            transforms.ToTensorV2()])
    
    train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
        if not args.no_cuda else {}

    valid_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
        if not args.no_cuda else {}
        
    # Datasets
    train_set = RC_Dataset(train_dataset, train_transform, args)
    valid_set = RC_Dataset(valid_dataset, valid_transform, args)

    # Dataloaders
    train_loader = data.DataLoader(train_set, **train_params)
    valid_loader = data.DataLoader(valid_set, **valid_params)
    
    return train_loader, valid_loader


# extract mean and std for images
def mean_std(args, X_list):
    image_path = args.data_path + '/images'
    mean = 0
    std = 0
    num = 0
    
    for trial in X_list:
        frame_list = os.listdir(os.path.join(image_path, trial))
        num += len(frame_list)
        
        for file in frame_list:
            file_path = os.path.join(image_path, trial, file)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_mean, img_std = cv2.meanStdDev(image)
            
            mean += img_mean / 255
            std += img_std / 255
            
    mean /= num
    std /= num
    
    mean = mean.reshape(-1).tolist()
    std = std.reshape(-1).tolist()
    
    print('mean : {:.3f},{:.3f},{:.3f}'.format(mean[0],mean[1],mean[2]))
    print('std : {:.3f},{:.3f},{:.3f}\n'.format(std[0],std[1],std[2]))

    return mean, std

# def show_sample(dataset,idx):
#     # params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': False}
#     # data_loader = data.DataLoader(dataset, **params)
    
#     image, target = dataset[idx]
#     seq_len = image.size(0)
    
#     mean = torch.tensor([0.387, 0.394, 0.545], dtype=torch.float32)
#     std = torch.tensor([0.184, 0.168, 0.171], dtype=torch.float32)
        
#     un_normalize = tf.Normalize((-mean/std).tolist(), (1/std).tolist())
    
#     fig=plt.figure(figsize=(12, 8))
    
#     for i in range(seq_len):
#         pos = '3{}{}'.format(seq_len,i+1)
#         pos = int(pos)
#         ax = plt.subplot(pos)
#         ax.imshow(un_normalize(image[i,...]).permute(1, 2, 0))
#         ax.set_title('t-{}'.format(seq_len-i))
#         ax.tick_params(left = False,
#                        bottom = False,
#                        labelbottom = False,
#                        labelleft = False)
        
#     ax = plt.subplot(323)
#     ax.plot(target[:,0])
#     ax.set_xticks(np.arange(seq_len))
#     ax.set_ylim([-1.2, 1.2])
#     ax.set_title('Steer X')
    
#     ax = plt.subplot(324)
#     ax.plot(target[:,1])
#     ax.set_xticks(np.arange(seq_len))
#     ax.set_ylim([-1.2, 1.2])
#     ax.set_title('Steer Y')
        
#     ax = plt.subplot(325)
#     ax.plot(target[:,2])
#     ax.set_xticks(np.arange(seq_len))
#     ax.set_ylim([-1.2, 1.2])
#     ax.set_title('Roll')
        
#     ax = plt.subplot(326)
#     ax.plot(target[:,3])   
#     ax.set_xticks(np.arange(seq_len))
#     ax.set_ylim([-1.2, 1.2])
#     ax.set_title('Feed')
        
#     plt.show()
# ## -------------------- (reload) model prediction ---------------------- ##
