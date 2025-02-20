import os 
import numpy as np 
from tqdm import tqdm 
import shutil 

old_dir = '/Volumes/My Passport for Mac/maxar-data/rene_senegal_trees/training_images'
new_dir = 'data'

file_names = os.listdir(old_dir)
np.random.shuffle(file_names)

for file_name in tqdm(file_names[0:150]):
    if 'annotation' in file_name and 'png' in file_name and 'aux' not in file_name:
        old_path = os.path.join(old_dir, file_name)
        new_path = os.path.join(new_dir, file_name)
        shutil.copy(old_path, new_path)

        file_number = file_name.split('_')[-1].split('.')[0]

        ndvi_name = f'ndvi_{file_number}.png'
        old_path = os.path.join(old_dir, ndvi_name)
        new_path = os.path.join(new_dir, ndvi_name)
        shutil.copy(old_path, new_path)

        pan_name = f'pan_{file_number}.png'
        old_path = os.path.join(old_dir, pan_name)
        new_path = os.path.join(new_dir, pan_name)
        shutil.copy(old_path, new_path)
