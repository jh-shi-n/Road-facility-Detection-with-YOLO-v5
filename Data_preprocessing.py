import os
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# os.chdir("./Hackerton/F1soft 119해커톤/")
path = os.getcwd()


## 데이터 시각화 _ Polygon
# 사용할 폴더 이름 설정
select_dir = str(input())

num_state = ["0","1"]

for j in num_state:
    temp = os.listdir("C:/python/Hackerton/F1soft 119해커톤/{}/{}/".format(select_dir, j))
    
    os.chdir("C:/python/Hackerton/F1soft 119해커톤/{}/{}/".format(select_dir, j))
    
    file_list_img = [file for file in temp if file.endswith(".jpg")]
    file_list_json= [file for file in temp if file.endswith(".json")]
    
    for i in range(len(file_list_img)):
        with open("{}.json".format(file_list_img[i][:-4]), "r", encoding = 'utf-8') as json_file:
            jdata = json.load(json_file)
            
            #Label 중 annotation 추출
            polyname = jdata['annotations'][0]['label_name']
            bbox_state = jdata['annotations'][0]['is_defect']
            
            label = jdata['annotations'][0]['label_id']
        
            img = cv2.imread(file_list_img[i])
            img_= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for j in range(len(jdata['annotations'])):
                polygon = jdata['annotations'][0]['annotation_info']
                img_p = cv2.polylines(img, [np.array(polygon, np.int32)],True, (255,0,255),8)
            plt.figure(figsize=(8,8))
            plt.imshow(img_p)
            plt.show()
            
            print("객체 이름 : {}, JSON_STATE : {}, **폴더 STATE** : {}".format(polyname,  bbox_state, num_state[j]))



## JSON 파일 내 annotation을 yolo 형식으로 변환
# 이미지 폴더 별 Json 파일을 일괄적으로 YOLO형식으로 변환처리
# 일부 파일의 state 상태가 비어있는 경우 & annotation이 비어있는 경우 제거
for i in folder:
    for j in group:
        path_group = path_d + i + '/' + j
        list_dir = os.listdir(path_group)
        list_jpg = [file for file in list_dir if file.endswith(".jpg")]
        list_json = [file for file in list_dir if file.endswith(".json")]
        os.makedirs(path_c + i + '/' + j )
        
        for k in range(len(list_json)):
            with open(path_d + i + '/' + j + '/' + list_json[k], "r", encoding = 'utf-8') as json_file:
                jdata = json.load(json_file)
                state = jdata['description']['state']
                
                if state is not None: 
                    label = jdata['annotations'][0]['label_id']
                    box = jdata['annotations'][0]['annotation_info'][0]
                    anno_type = jdata['annotations'][0]['annotation_type']
                    img = cv2.imread(path_d + i + '/' + j + '/' + list_json[k].split('.')[0] + '.jpg')
                    
                    if (img is not None) and (anno_type == "bbox"):
                        shape = img.shape
                        x = box[0]
                        y = box[1]
                        w = box[2]
                        h = box[3]
                        list_json[k].split('.')[0] + '.jpg'
                        
                        #변환 작업
                        x_center = str((x + w/2) / shape[1])
                        y_center = str((y + h/2) / shape[0])
                        scale_w = str(w / shape[1])
                        scale_h = str(h / shape[0])
                        
                        f = open(path_c + i + '/' + j + '/' + list_json[k].split('.')[0] + '.txt', "w")
                        f.write(label + ' ' + x_center + ' ' + y_center + ' ' + scale_w + ' ' + scale_h)
                        f.close()


## 이미지 Resizing
# 전체 이미지를 학습 및 이동속도가 빠르도록 (416 X 416) 형태로 일괄 변환

import io
from PIL import Image, ImageOps


before_path = "C:/python/Hackerton/작업후"
after_path = "C:/python/Hackerton/resizing"

folder_names = ['01', '02','03','04','05', '06', '07', '08', '09', '10', '11', '12',
 '13', '14', '15', '16', '21', '23', '24', '25', '26', '27', '28', '29', '30', '31',
 '32', '36', '37', '38']

folder_state = ['0','1']

for i in range(len(folder_names)):
    for j in range(0,2):
        file_names = os.listdir("{}/{}/{}/".format(before_path, folder_names[i], folder_state[j]))
        jpg_file = [word for word in file_names if '.jpg' in word]
        for k in range(len(jpg_file)):
            original_image = Image.open("{}/{}/{}/{}".format(before_path, folder_names[i], folder_state[j], jpg_file[k]))
            fixed_image = ImageOps.exif_transpose(original_image)

            fin = fixed_image.resize((416,416))

            fin.save("{}/{}/{}/{}".format(after_path, folder_names[i], folder_state[j], jpg_file[k]))
            
    print("완료 : {} 폴더".format(i))

