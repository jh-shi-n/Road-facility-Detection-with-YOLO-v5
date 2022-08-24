
### change log (2022.08.24)
- fix typo in the readme
---

# 도로시설물 이상 탐지 여부
- 활용 OpenSource : Yolo V5 (https://github.com/ultralytics/yolov5)
- 활용 Pre-Trained Model : Yolo V5s
- 데이터 : 비공개 데이터(도로 시설물 이미지 데이터 및 JSON파일)
- 해커톤 제출 양식에 따라 detect.py 코드 수정 

## 데이터 전처리 (Data_preprocessing.py)
- 이미지 데이터 사이즈 (416X416) 으로 수정
- 함께 제공된 JSON 파일의 annotation 형태에 따라 다르게 수정
- BBox(Bounding Box) 일 경우, json에서 bbox 수치만 추출 후, normalized
- Polygon 일 경우, 이미지에 Polygon 형태 시각화 후, 해당 형태에 최대한 맞춰 BBox 직접 생성 (labelimg 사용)

## 도로 시설물 이미지 Detect 결과 도출 (Detect.py)
- 검출 결과에 대한 summary.txt 파일 생성 (기존 코드 내 일부 수정)
- 각 파일에 대한 검출 결과에 대한 txt 파일 생성 (기본 옵션)
