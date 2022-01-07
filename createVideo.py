import cv2
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split

prediction_path = './predictions'
video_folder = 'videos'
video_path = os.path.join(prediction_path, video_folder)
os.makedirs(video_path, exist_ok=True)
    
data_path = "/Users/217/RC/dataset" 

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

for fname in valid_image_list:
    
    img_array = []
    for filename in glob.glob(os.path.join(image_folder, fname, '*')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    
    out = cv2.VideoWriter(os.path.join(video_path,'{}.avi'.format(fname)),cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
        
    out.release()