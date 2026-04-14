"""
YOLOv8 (Swin Backbone) .pt -> TensorRT FP16 엔진 변환 스크립트
- 반드시 Jetson Orin Nano 위에서 실행해야 합니다 (TensorRT 엔진은 디바이스 종속적)
- 실행 전 필요 패키지: ultralytics, tensorrt, torch, timm

사용법:
    python export_tensorrt.py                          # 기본값 사용
    python export_tensorrt.py --weights best.pt        # 가중치 지정
    python export_tensorrt.py --imgsz 640 --workspace 4  # 옵션 변경
"""

import argparse
import torch
import torch.nn as nn
import timm
from ultralytics import YOLO


# --- 커스텀 Swin 백본 클래스 (best.pt 로딩에 필요) ---
class SwinStage0(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=False, features_only=True
        )
        self.p4 = None
        self.p5 = None

    def forward(self, x):
        features = self.swin(x)
        self.p4 = features[2]
        self.p5 = features[3]
        return features[1]


class SwinStage1(nn.Module):
    def __init__(self, stage0):
        super().__init__()
        self.stage0 = stage0

    def forward(self, x):
        return self.stage0.p4


class SwinStage2(nn.Module):
    def __init__(self, stage0):
        super().__init__()
        self.stage0 = stage0

    def forward(self, x):
        return self.stage0.p5


def export(weights: str, imgsz: int, workspace: int):
    print(f"[1/3] 모델 로딩: {weights}")
    model = YOLO(weights)

    print(f"[2/3] TensorRT FP16 엔진 변환 시작 (imgsz={imgsz}, workspace={workspace}GB)")
    export_path = model.export(
        format="engine",
        half=True,           # FP16
        imgsz=imgsz,
        workspace=workspace,  # TensorRT 빌더 최대 워크스페이스 (GB)
        device=0,
        simplify=True,        # ONNX simplify 후 변환
    )
    print(f"[3/3] 변환 완료: {export_path}")
    print("Jetson에서 추론 예시:")
    print(f'    model = YOLO("{export_path}")')
    print('    results = model.predict(source="image.jpg", imgsz=640)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 .pt -> TensorRT FP16 엔진 변환")
    parser.add_argument(
        "--weights", type=str,
        default="runs/detect/train/weights/best.pt",
        help="변환할 .pt 가중치 경로",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="입력 이미지 크기 (학습 시 사용한 크기와 동일하게)",
    )
    parser.add_argument(
        "--workspace", type=int, default=4,
        help="TensorRT 빌더 워크스페이스 (GB). Orin Nano 8GB 기준 4 권장",
    )
    args = parser.parse_args()
    export(args.weights, args.imgsz, args.workspace)
