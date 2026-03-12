import torch
import torch.nn as nn
import timm
from ultralytics import YOLO

# --- 커스텀 어댑터: Swin의 뇌를 YOLO의 목(Neck)에 분배하는 장치 ---
class SwinStage0(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=True)
        self.swin.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        
        # [핵심] 기존 15에폭 동안 똑똑해진 뇌가 망가지지 않도록 얼림(Freeze)
        for param in self.swin.parameters():
            param.requires_grad = False  

        self.p4 = None
        self.p5 = None

    def forward(self, x):
        features = self.swin(x)
        self.p4 = features[2] # 384 차원 (P4 저장)
        self.p5 = features[3] # 768 차원 (P5 저장)
        return features[1]    # 192 차원 (P3 배출)

class SwinStage1(nn.Module):
    def __init__(self, stage0):
        super().__init__()
        self.stage0 = stage0
    def forward(self, x):
        return self.stage0.p4 # 저장된 P4 배출

class SwinStage2(nn.Module):
    def __init__(self, stage0):
        super().__init__()
        self.stage0 = stage0
    def forward(self, x):
        return self.stage0.p5 # 저장된 P5 배출

def inject_swin(model, weight_path):
    print("🧠 YOLOv8 몸통에 Swin Backbone(DINO) 이식 수술을 시작합니다...")
    s0 = SwinStage0(weight_path)
    s1 = SwinStage1(s0)
    s2 = SwinStage2(s0)

    # YOLO 내부 루프 속이기
    s0.i, s0.f, s0.type = 0, -1, 'SwinStage0'
    s1.i, s1.f, s1.type = 1, -1, 'SwinStage1'
    s2.i, s2.f, s2.type = 2, -1, 'SwinStage2'

    # 몸통 강제 결합
    model.model.model[0] = s0
    model.model.model[1] = s1
    model.model.model[2] = s2
    print("✅ 수술 대성공! 완벽하게 결합되었습니다.")
    return model

def main():
    # 1. 방금 만든 커스텀 껍데기(yaml) 불러오기
    model = YOLO('yolov8_swin.yaml')

    # 2. 1단계에서 15에폭 완성한 가중치 파일 이름 입력 (🚨 정확한 파일명 확인!)
    WEIGHT_PATH = 'swin_dino_ep15.pth' 
    model.model = inject_swin(model.model, WEIGHT_PATH)

    # 3. 본격적인 학습 시작!
    print("🔥 YOLOv8 파인튜닝을 시작합니다...")
    model.train(
        # 🚨 데이터셋 폴더 경로 (이전에 만드신 data.yaml의 절대 경로 입력)
        data='/home/user-511/dataset/data.yaml', 
        epochs=50,      # 파인튜닝은 50에폭 정도면 충분합니다
        imgsz=224,      # Swin 모델 고정 해상도
        batch=16,
        device=0,       # 듀얼 GPU 병렬 충돌(Pickle Error) 방지를 위해 GPU 0번만 사용 (백본을 얼려둬서 1개로도 엄청 빠릅니다)
        optimizer='AdamW',
        lr0=0.001
    )

if __name__ == '__main__':
    main()
