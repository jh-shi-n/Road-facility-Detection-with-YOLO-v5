import argparse
import cv2
import json
import os

def run(opt) :
    ## JSON 파일 내 annotation을 yolo 형식으로 변환
    ## 이미지 폴더 별 Json 파일을 일괄적으로 YOLO형식으로 변환처리
    ## 일부 파일의 state 상태가 비어있는 경우 & annotation이 비어있는 경우 제거
    for i in opt.class_folder:
        for j in opt.class_name:
            path_group = opt.path_origin + i + '/' + j
            list_dir = os.listdir(path_group)
            list_jpg = [file for file in list_dir if file.endswith(".jpg")]
            list_json = [file for file in list_dir if file.endswith(".json")]
            os.makedirs(opt.path_yolo + i + '/' + j )
            
            for k in range(len(list_json)):
                with open(opt.path_origin + i + '/' + j + '/' + list_json[k], "r", encoding = 'utf-8') as json_file:
                    jdata = json.load(json_file)
                    state = jdata['description']['state']
                    
                    if state is not None: 
                        label = jdata['annotations'][0]['label_id']
                        box = jdata['annotations'][0]['annotation_info'][0]
                        anno_type = jdata['annotations'][0]['annotation_type']
                        img = cv2.imread(opt.path_origin + i + '/' + j + '/' + list_json[k].split('.')[0] + '.jpg')
                        
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
                            
                            f = open(opt.path_yolo + i + '/' + j + '/' + list_json[k].split('.')[0] + '.txt', "w")
                            f.write(label + ' ' + x_center + ' ' + y_center + ' ' + scale_w + ' ' + scale_h)
                            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_origin', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--path_yolo', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--class_folder', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--class_name', type=float, default=0.25, help='confidence threshold')
    opt = parser.parse_args()

    run(opt)