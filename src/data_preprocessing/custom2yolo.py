import argparse
import cv2
import json
import os

def run(opt) :
    ## Custom JSON 파일 정보 전처리 후 yolo 형식으로 변환
    ## 일부 파일의 state 상태가 비어있는 경우 & annotation이 비어있는 경우 제거
    for i in opt.class_folder:
        for j in opt.class_name:
            path_group = os.path.join(f"{opt.path_origin}i", j)
            list_dir = os.listdir(path_group)
            list_jpg = [file for file in list_dir if file.endswith(".jpg")]
            list_json = [file for file in list_dir if file.endswith(".json")]
            os.makedirs(f"{opt.path_yolo}i", j)
            
            for k in range(len(list_json)):
                file_name = list_json[k].split(".")[0]
                origin_file_path_format = os.path.join(f"{opt.path_origin}i", j, file_name)
                yolo_file_path_format = os.path.join(f"{opt.path_yolo}i", j, file_name)
                with open(f"{file_path_format}.json", "r", encoding = 'utf-8') as json_file:
                    jdata = json.load(json_file)
                    state = jdata['description']['state']
                    
                    if state is not None: 
                        label = jdata['annotations'][0]['label_id']
                        box = jdata['annotations'][0]['annotation_info'][0]
                        anno_type = jdata['annotations'][0]['annotation_type']
                        img = cv2.imread(f"{file_path_format}.jpg")
                        
                        if (img is not None) and (anno_type == "bbox"):
                            shape = img.shape
                            x = box[0]
                            y = box[1]
                            w = box[2]
                            h = box[3]
                            
                            #변환 작업
                            x_center = str((x + w/2) / shape[1])
                            y_center = str((y + h/2) / shape[0])
                            scale_w = str(w / shape[1])
                            scale_h = str(h / shape[0])

                            with open(f"{yolo_file_path_format}.txt", "w") as f:
                                f.write(f"{label} {x_center} {y_center} {scale_w} {scale_h}")
                            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_origin', type=str, require=True)
    parser.add_argument('--path_yolo', type=str, require=True)
    parser.add_argument('--class_folder', type=str, require=True)
    parser.add_argument('--class_name', type=str, require=True)
    opt = parser.parse_args()

    run(opt)
