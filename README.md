# Private Data _ 도로시설물 이상 탐지 여부

- 활용 Repo : Yolo V5 (https://github.com/ultralytics/yolov5)
- 제출 양식에 따라 detect.py  수정 (txt 형식 추출 +  summary 저장)

## 데이터 전처리
- 이미지 사이즈 416으로 수정
- 함께 제공된 JSON 파일의 annotation 형태에 따라 다르게 수정
- BBox(Bounding Box) 일 경우, json에서 bbox 수치만 추출 후, normalized
- Polygon 일 경우, 이미지에 Polygon 형태 시각화 후, 해당 형태에 최대한 맞춰 BBox 직접 생성 (labelimg 사용)
