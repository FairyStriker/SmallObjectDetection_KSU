# src/

Swin-DINO + YOLOv8 소형 객체 탐지 파이프라인 소스 코드.

| 파일 | 역할 |
|------|------|
| `model.py` | Swin-Tiny 백본 + DINO Head 모델 정의 (`MultiCropSwinDINO`) |
| `data_aug.py` | DINO 자기지도학습용 Multi-Crop 데이터 증강 파이프라인 |
| `train.py` | 1단계: DINO 자기지도학습 훈련 루프 (Swin 백본 사전학습) |
| `train_swin_yolo.py` | 2단계: 사전학습된 Swin 백본을 YOLOv8에 이식 후 파인튜닝 |
| `export_tensorrt.py` | 학습된 best.pt를 TensorRT FP16 엔진으로 변환 (Jetson Orin Nano용) |
| `yolov8_swin.yaml` | Swin 백본 적용된 YOLOv8 모델 구조 설정 |
