# -*- coding: utf-8 -*-
import os
import cv2

dir_name = './/test2017//test2017'

file_list = os.listdir(dir_name)

out_file = './/out_img'

idx = 0

if not os.path.exists(out_file):
    os.mkdir(out_file)

for path in file_list:
    
    img = cv2.imread(dir_name + "//" + path)
    
    h = img.shape[0]
    w = img.shape[1]
    
    if h > 0.9*w and w > 0.9*h:
        tmp_path = out_file + "//" + str(idx) + ".jpg"
        while os.path.exists(tmp_path):
            idx += 1
            tmp_path = out_file + "//" + str(idx) + ".jpg"
        cv2.imwrite(tmp_path, img)
        idx += 1
        