import torch
import torch.nn as nn
import timm
from ultralytics import YOLO

# 🚨 [중요] 저장된 best.pt를 정상적으로 불러오기 위해, 
# 학습할 때 사용했던 커스텀 뼈대 클래스들을 여기에 반드시 명시해야 합니다.
class SwinStage0(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=True)
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

def main():
    # 1. 학습된 가중치 경로 (보통 runs/detect/train/weights/best.pt 에 저장됩니다)
    # 🚨 학습 중이라면 'best.pt' 대신 'last.pt'를 넣어서 중간 점검도 가능합니다!
    MODEL_PATH = "runs/detect/train/weights/best.pt" 
    
    # 2. 테스트해볼 이미지 경로 (16만 장 중 아무거나, 혹은 아까 쓰셨던 test_image.jpg)
    IMAGE_PATH = "test_image.jpg" 

    print(f"🚀 뇌 이식 완료된 커스텀 모델({MODEL_PATH})을 불러옵니다...")
    model = YOLO(MODEL_PATH)

    print(f"📸 '{IMAGE_PATH}' 이미지에서 작은 객체 탐지를 시작합니다...")
    
    # 3. 모델 추론 (바운딩 박스 그리기)
    results = model.predict(
        source=IMAGE_PATH,
        imgsz=224,      # 학습 시 고정했던 224 해상도 유지
        conf=0.25,      # 신뢰도 25% 이상인 박스만 그리기 (너무 안 잡히면 0.1로 내리고, 너무 많이 잡히면 0.5로 올리세요)
        iou=0.45,       # 겹치는 박스 제거 비율
        save=True,      # 결과를 이미지 파일로 저장
        show=False      # 우분투 터미널 환경을 고려해 화면 팝업은 끔 (GUI 환경이면 True로 변경 가능)
    )

    print("\n✅ 탐지 완료!")
    print("결과 이미지는 'runs/detect/predict/' (혹은 predict2, predict3...) 폴더 안에 저장되었습니다.")
    print("해당 폴더로 가셔서 바운딩 박스가 쳐진 이미지를 확인해 보세요!")

if __name__ == '__main__':
    main()
