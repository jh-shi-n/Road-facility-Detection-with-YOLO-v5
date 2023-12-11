import os
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageOps

## 데이터 시각화 _ Polygon
# 사용할 폴더 이름 설정
def viz_dataset():
    select_dir = str(input())
    num_state = ["0","1"]

    for j in num_state:
        file_path_set = "FILE_PATH/{}/{}".format(select_dir, j)
        temp = os.listdir(file_path_set)
        
        file_list_img = [file for file in temp if file.endswith(".jpg")]
        file_list_json= [file for file in temp if file.endswith(".json")]
        
        for i in range(len(file_list_img)):
            get_file_name = file_list_img[i].split(".")[0]
            with open(os.path.join(file_path_set, f"{get_file_name}.json").format(), "r", encoding = 'utf-8') as json_file:
                json_data = json.load(json_file)
                
                # Label 중 annotation 추출
                polyname = json_data['annotations'][0]['label_name']
                bbox_state = json_data['annotations'][0]['is_defect']
                label = json_data['annotations'][0]['label_id']
            
                img = cv2.imread(file_list_img[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                for j in range(len(json_data['annotations'])):
                    polygon = json_data['annotations'][0]['annotation_info']
                    img_p = cv2.polylines(img, [np.array(polygon, np.int32)],True, (255,0,255),8)
                plt.figure(figsize=(8,8))
                plt.imshow(img_p)
                plt.show()
                
                print(f"객체 이름 : {polyname}, JSON_STATE : {bbox_state}, **폴더 STATE** : {num_state[j]}")


## 이미지 Resizing
## 모델 학습 진행 시 시간 절약을 위해 (416 X 416) 형태로 일괄 변환
def image_resizing():
    before_path = "FILE_PATH_ORIGIN"
    after_path = "FILE_PATH_RESIZE"

    folder_names = [f'{x:>02}' for x in range(0,39)]
    folder_state = ['0','1']

    for i in range(len(folder_names)):
        for j in range(0,2):
            file_names = os.listdir("{}/{}/{}/".format(before_path, folder_names[i], folder_state[j]))
            jpg_file = [word for word in file_names if '.jpg' in word]
            for k in range(len(jpg_file)):
                original_image = Image.open("{}/{}/{}/{}".format(before_path, folder_names[i], folder_state[j], jpg_file[k]))
                fixed_image = ImageOps.exif_transpose(original_image)
                fin = fixed_image.resize((416,416))

                fin.save(f"./{after_path}/{ folder_names[i]}/{folder_state[j]}/{jpg_file[k]}")
                
        print(f"완료 : {i} 폴더")

