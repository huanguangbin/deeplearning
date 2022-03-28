import os
import cv2
import numpy as np
from tqdm import tqdm

files_dir = r"D:\tmp\data_pix2pix\GAN_G\compose_results\small_size_elastic\test1"
save_dir = r"D:\tmp\data_pix2pix\GAN_G\compose_results\elastic\test"
files_name = os.listdir(files_dir)
for file_name in tqdm(files_name):
    file_path = os.path.join(files_dir, file_name)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    concat_img = np.concatenate((img, img), axis=1)
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, concat_img)
