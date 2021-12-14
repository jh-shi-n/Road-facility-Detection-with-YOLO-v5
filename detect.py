# -*- coding: utf-8 -*-
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
사용방식:
    $ python path/to/detect.py --weights yolov5s.pt --source path/  # directory
"""

import argparse
import os
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
SAVE_ROOT = Path(os.getcwd())

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT ,   # 결과 저장 directory (1)
        name='test_output',  # 결과 저장 directory (2)
        exist_ok=True,  
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,
        hide_conf=False, 
        half=False, 
        dnn=False,  
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # 저장 Dictionary _ 탐지 결과 및 객체 이름 생성
    save_dir = Path(SAVE_ROOT) / name
    if os.path.isdir(save_dir) is False:  # 폴더 생성 시
      save_dir.mkdir(parents=True, exist_ok=True)

    # Model 로드
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, pt, jit, onnx, engine = model.stride,  model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = ['자동차진입_억제용_말뚝_스테인리스', '자동차진입_억제용_말뚝_스테인리스_불량', '자동차진입_억제용_말뚝_탄성고무', '자동차진입_억제용_말뚝_탄성고무_불량', '시선유도봉_2줄', '시선유도봉_2줄_불량', '시선유도봉_3줄', '시선유도봉_3줄_불량', '보행자용_방호울타리', '보행자용_방호울타리_불량', '보행자용_방호울타리_불량부분', '교량용_방호울타리', '교량용_방호울타리_불량', '교량용_방호울타리_불량부분', '턱낮추기', '턱낮추기_불량', '턱낮추기_불량부분', '경사로', '경사로_불량', '점자블럭', '점자블럭_불량', '점자블럭_불량부분', '도로안내표지_지주', '도로안내표지_지주_불량', '시멘트_콘크리트', '시멘트_콘크리트_불량', '시멘트_콘크리트_불량부분', '보도블록', '보도블록_불량', '보도블록_불량부분', '자전거도로', '자전거도로_불량', '자전거도로_불량부분', '연석', '연석_불량', '연석_불량부분', '무단횡단_방지_울타리', '무단횡단_방지_울타리_불량', '무단횡단_방지_울타리_불량부분', '맨홀', '맨홀_불량', '현장신호제어기', '현장신호제어기_불량', '시각장애인용_음향신호기', '시각장애인용_음향신호기_불량', '과속방지턱', '과속방지턱_불량', '과속방지턱_불량부분', '횡단보도', '횡단보도_불량', '횡단보도_불량부분', '고원식횡단보도', '고원식횡단보도_불량', '고원식횡단보도_불량부분', '통합표지판', '통합표지판_불량' , '정주식_부착식_표지', '정주식_부착식_표지_불량', '가로등', '가로등_불량', '전봇대_빗금표시', '전봇대_빗금표시_불량', '자동차진입_억제용_말뚝_대리석', '자동차진입_억제용_말뚝_대리석_불량', '자동차진입_억제용_말뚝_U자형', '자동차진입_억제용_말뚝_U자형_불량', '소화전', '소화전_불량', '주차멈춤턱_블럭형', '주차멈춤턱_블럭형_불량', '미끄럼방지계단', '미끄럼방지계단_불량']
    class_normalcode = [0, 2, 4 ,6 ,8 , 11, 14, 17, 19, 22, 24, 27, 30, 33, 36, 39, 41, 43, 45, 48, 51, 54, 56, 58, 60, 62, 64]

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # 저장 Directory 준비 _ Summary 관련
    save_summary_dir = Path(SAVE_ROOT) / "test_summary"
    if os.path.isdir(save_summary_dir) is False:  # 폴더 생성 시
      save_summary_dir.mkdir(parents=True, exist_ok=True)

    # 탐지 시작
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # 예측
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS _ option (1)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # 예측 과정 및 BBOX 추가
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            img_h, img_w = im0.shape[0:2]

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir  / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Bbox 사이즈를 현재 이미지 사이즈에 맞게 조정
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 결과 출력 _ 이미지명, 객체명, 갯수 Summary
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                   

                # 결과 출력 _ 이미지 별 객체 및 bbox 
                for *xyxy, conf, cls in reversed(det):
                    state_zero = ["0 "]
                    state_one = ["1 "]
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        # Yolo BBox result 추출
                        class_num = str(int(line[0]))
                        xmin_value = int(xyxy[0])
                        ymin_value = int(xyxy[1])
                        xmax_value = int(xyxy[2])
                        ymax_value = int(xyxy[3])

                        #파일 판단 ~ 정상인 경우 0 , 비정상인 경우 1
                        if int(class_num) in class_normalcode:
                            state_zero.append("{} : [{}, {}, {}, {}]".format(class_num.zfill(2),xmin_value ,ymin_value ,xmax_value , ymax_value)+"\n")
                        else:
                            state_one.append("{} : [{}, {}, {}, {}]".format(class_num.zfill(2),xmin_value ,ymin_value ,xmax_value , ymax_value)+"\n")


                        with open(txt_path + '.txt', 'a') as f:
                            if len(state_zero) > 1:
                                for i in range(len(state_zero)):
                                    f.write(state_zero[i])
                            elif len(state_one) > 1:
                                for z in range(len(state_one)):
                                    f.write(state_one[z])


        # Summary 파일 내 결과 입력
        with open(str(save_summary_dir) + "/" + 'test_summary.txt', 'a') as sum_f:
            sum_f.write(s + "\n")

    # 최종 결과 print
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if save_txt or save_img:
        LOGGER.info(f"파일별 이상 탐지 txt 결과를 저장하였습니다 : {colorstr('bold', save_dir)}")
        LOGGER.info(f"전체 파일 탐지 Summary 결과를 저장하였습니다 : {colorstr('bold', save_summary_dir)}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT , help='save results to project/name')
    parser.add_argument('--name', default='test_output', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)