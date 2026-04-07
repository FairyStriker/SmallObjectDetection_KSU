# Small Object Detection — Swin · DINO · YOLOv8

> **Swin Transformer 백본**을 **DINO 자기지도학습**으로 16만 장 비라벨 이미지에서 사전학습한 뒤,
> **YOLOv8** 탐지 프레임워크에 이식하여 **10개 클래스 소형 객체**를 탐지하는 2단계 학습 파이프라인입니다.

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF?logo=yolo&logoColor=black" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Backbone-Swin--Tiny-7B68EE" alt="Swin">
  <img src="https://img.shields.io/badge/SSL-DINO-FF6F61" alt="DINO">
  <img src="https://img.shields.io/badge/Status-Training%20in%20Progress-orange" alt="Status">
  <img src="https://img.shields.io/badge/Classes-10-blue" alt="Classes">
</p>

---

## 📊 Results — 학습 결과

> 🚧 **현재 `imgsz=640` 으로 재학습 진행 중입니다.**
> 학습이 완료되는 대로 본 섹션에 전체 / 클래스별 mAP 결과가 추가될 예정입니다.

### Overall (Validation)

| Metric        | Value |
|---------------|-------|
| Precision     | _TBD_ |
| Recall        | _TBD_ |
| mAP@0.5       | _TBD_ |
| mAP@0.5–0.95  | _TBD_ |

### Per-class Performance

| Class           | Precision | Recall | mAP@0.5 | mAP@0.5–0.95 |
|-----------------|-----------|--------|---------|--------------|
| Fishing_Boat    | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Merchant_Ship   | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Warship         | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Person          | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Bird            | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Fixed_Wing      | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Rotary_Wing     | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| UAV             | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Leaflet         | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Trash_Bomb      | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 🎯 Goal — 프로젝트 목표

작은 객체(Small Object)는 픽셀이 적어 일반 탐지기로는 놓치기 쉽습니다.
이 프로젝트는 **CNN 백본의 한계(국소 수용영역)** 를 **Transformer 백본 + 자기지도 사전학습** 으로 보완해
**소형 객체 탐지 정확도**를 끌어올리는 것을 목표로 합니다.

```
[기존 YOLOv8]
입력 → CSPDarknet (CNN) → FPN Neck → Detect Head → 결과

[본 프로젝트]
입력 → Swin Transformer (DINO 사전학습) → FPN Neck → Detect Head → 결과
```

---

## 🧠 Why Swin + DINO?

| 항목 | CSPDarknet (기존) | Swin Transformer (본 프로젝트) |
|---|---|---|
| 연산 방식 | 로컬 컨볼루션 | Shifted Window Attention |
| 수용 영역 | 레이어가 깊어질수록 점진적 확장 | 전역 문맥을 직접 파악 |
| 사전학습 | ImageNet 분류 (1,000 클래스 라벨 필요) | **DINO 자기지도학습 (라벨 없는 16만 장)** |
| 소형 객체 | 상대적으로 취약 | **로컬 크롭 학습으로 강화** |

DINO는 분류 레이블 없이 **이미지 내 구조적 패턴**을 스스로 학습합니다.
로컬 크롭(scale 0.05–0.4)으로 작은 영역을 확대 학습하기 때문에,
일반적인 ImageNet 분류 사전학습보다 **소형 객체의 경계와 텍스처에 더 민감한 특징**을 형성합니다.

---

## 🔄 Pipeline — 전체 파이프라인

```
[ 라벨 없는 이미지 161,381장 ]
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 · DINO 자기지도 사전학습 (src/train.py)         │
│                                                         │
│  멀티크롭 증강 (Global×2 + Local×6)                     │
│      ├── Student (Swin + DINOHead) ──┐                  │
│      └── Teacher (Swin + DINOHead, EMA) ──┐             │
│                                            │            │
│           ┌────── DINO Loss ───────────────┘            │
│           ▼                                             │
│       Student 역전파 → Teacher EMA 업데이트              │
│                                                         │
│  ✓ 15 에폭 학습 후 Swin 백본 가중치만 저장               │
│    (DINOHead는 임시 교사 역할 후 폐기)                   │
└─────────────────────────────────────────────────────────┘
            │  swin_dino_ep15.pth
            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 · YOLOv8 파인튜닝 (src/train_swin_yolo.py)      │
│                                                         │
│  Swin 백본 이식 (동결) → FPN Neck → Detect Head          │
│  ※ Neck + Head만 학습, 백본은 건드리지 않음              │
└─────────────────────────────────────────────────────────┘
            │
            ▼
[ 10개 클래스 소형 객체 탐지기 ]
```

---

## 🏗️ Stage 1 — DINO Self-Supervised Pretraining

### 핵심 아이디어

DINOHead는 Swin 백본이 좋은 특징을 뽑도록 **학습 신호를 주는 임시 도구**입니다.
학습이 끝나면 DINOHead는 폐기하고, **Swin 백본 가중치만** 저장합니다.

> 비유하자면 DINOHead는 Swin을 가르치는 **임시 선생님**이고,
> 졸업 후에는 필요 없습니다. 중요한 것은 Swin에 녹아든 **"잘 보는 능력"** 입니다.

### Multi-Crop Augmentation (`src/data_aug.py`)

한 장의 이미지에서 **8개의 크롭**을 동시에 생성합니다.

| 크롭 종류 | 개수 | 출력 크기 | 원본 스케일 범위 | 목적 |
|---|---:|---|---|---|
| Global View | 2장 | 224×224 | 0.4 ~ 1.0 | 전체 맥락 파악 |
| Local View | 6장 | 224×224 | **0.05 ~ 0.4** | **소형 객체 확대 학습** |

`local_size=224`로 작은 영역을 224 해상도까지 확대해서 보는 것이 **소형 객체 탐지 성능을 높이기 위한 핵심 설계 선택** 입니다.

### Student–Teacher Architecture

```
이미지 8크롭 → Student (전체 8장 처리) ─┐
이미지 2크롭 → Teacher (글로벌 2장만) ─┤→ DINO Loss → Student만 역전파
                  ↑                    │
                EMA 업데이트 ←──────────┘
              (momentum = 0.996)
```

- **Student** : 역전파로 직접 학습
- **Teacher** : Student의 EMA(지수이동평균)로만 갱신 (gradient 없음)
- Teacher가 더 안정적인 표현을 유지하므로, Student가 Teacher를 모방하면서 고품질 특징을 학습합니다.

### Mode-Collapse Defense

DINO 학습에서 가장 큰 위험은 **모드 콜랩스**(모든 입력이 같은 출력으로 수렴)입니다.
세 가지 메커니즘으로 방지합니다.

```python
# 1. Centering — Teacher 출력의 평균을 빼서 특정 차원의 지배를 억제
teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)

# 2. Sharpening — teacher_temp = 0.07로 분포를 날카롭게 유지

# 3. Gradient Clipping — 기울기 폭발 방지
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
```

추가로 `lr = 5e-5`의 극도로 낮은 학습률을 사용해 **사전학습된 Swin 가중치가 파괴되지 않도록** 보호합니다.

### Dual-GPU Support

```python
if torch.cuda.device_count() > 1:
    student = nn.DataParallel(student)
    teacher = nn.DataParallel(teacher)
```

`torch.amp.autocast` + `GradScaler`로 혼합 정밀도(AMP) 학습을 사용하며,
듀얼 Quadro P5000 환경에서 16만 장 × 15 에폭을 학습했습니다.

### Weight Saving

```python
# DINOHead는 버리고 Swin 백본 가중치만 저장
torch.save(student.module.backbone.state_dict(), f"swin_dino_ep{epoch+1}.pth")
```

---

## 🎯 Stage 2 — YOLOv8 Fine-tuning

### 핵심 아이디어 — 백본 이식 수술

`yolov8_swin.yaml`의 `backbone` 섹션은 **자리만 잡아주는 더미 Conv 레이어**입니다.
`inject_swin()` 함수가 이 자리를 실제 Swin Transformer로 **물리적으로 교체**합니다.

### YAML Dummy Backbone (`src/yolov8_swin.yaml`)

```yaml
backbone:
  - [-1, 1, Conv, [192, 3, 2]]  # 0: 더미 → SwinStage0 (P3, 192ch)
  - [-1, 1, Conv, [384, 3, 2]]  # 1: 더미 → SwinStage1 (P4, 384ch)
  - [-1, 1, Conv, [768, 3, 2]]  # 2: 더미 → SwinStage2 (P5, 768ch)
```

### Adapter Pattern

Swin은 **하나의 forward**에서 P3·P4·P5를 동시에 계산합니다.
YOLOv8은 레이어를 순차 호출하므로 다음 어댑터로 변환합니다.

```
SwinStage0 : Swin 실제 연산 수행
             ├── P3 (192ch) → 즉시 출력 (YOLO 레이어 0)
             ├── P4 (384ch) → self.p4에 캐시
             └── P5 (768ch) → self.p5에 캐시

SwinStage1 : self.stage0.p4 반환 (YOLO 레이어 1)
SwinStage2 : self.stage0.p5 반환 (YOLO 레이어 2)
```

```python
def inject_swin(model, weight_path):
    s0 = SwinStage0(weight_path)  # 실제 Swin 연산 + 가중치 로드
    s1 = SwinStage1(s0)           # P4 어댑터
    s2 = SwinStage2(s0)           # P5 어댑터

    model.model[0] = s0           # 더미 Conv 교체
    model.model[1] = s1
    model.model[2] = s2
```

### Backbone Channel Comparison

| 특징맵 | CSPDarknet (기존) | Swin Transformer (본 프로젝트) |
|---|---:|---:|
| P3 (소형 객체) | 256 ch | **192 ch** |
| P4 (중형 객체) | 512 ch | **384 ch** |
| P5 (대형 객체) | 1024 ch | **768 ch** |

Swin의 계층적 구조 덕분에 P3/P4/P5가 자연스럽게 생성되며,
YOLO의 FPN Neck과 채널 수가 맞도록 YAML을 설계했습니다.

### Training Loop

```python
model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,         # 파인튜닝 입력 해상도
    batch=8,
    device=0,
    optimizer='AdamW',
    lr0=0.0001         # Neck/Head 학습률 (백본은 동결)
)
```

Swin 백본은 `requires_grad = False`로 동결되어 **Neck과 Detect Head만 학습**됩니다.

---

## 📁 Project Structure — 파일 구조

```
SmallObjectDetection_KSU/
├── README.md                  ← 본 문서
├── .gitattributes             ← Git LFS 설정
├── .gitignore
│
├── src/                       ← 학습 / 추론 코드
│   ├── model.py               ·  MultiCropSwinDINO (Swin + DINOHead)
│   ├── data_aug.py            ·  멀티크롭 증강 (Global ×2 + Local ×6)
│   ├── train.py               ·  Stage 1 — DINO 자기지도 사전학습
│   ├── train_swin_yolo.py     ·  Stage 2 — YOLOv8 파인튜닝 + 백본 이식
│   ├── yolov8_swin.yaml       ·  YOLOv8 커스텀 구조 (더미 백본 + Neck/Head)
│   ├── sanity_check.py        ·  멀티크롭 + 모델 동작 확인
│   └── test_model.py          ·  학습된 best.pt 추론 테스트
│
└── logs/                      ← 학습 로그 (재현 검증용)
    ├── pretrain.log           ·  Stage 1 DINO 사전학습 로그 (15 epoch)
    └── finetune.log           ·  Stage 2 YOLOv8 파인튜닝 로그 (50 epoch)
```

---

## ⚙️ How to Run — 실행 방법

### 1. 환경 설치

```bash
pip install torch torchvision timm ultralytics
```

> 권장 : Python 3.8+, CUDA 11.8, PyTorch 2.4

### 2. Stage 1 — DINO 사전학습

```bash
# src/train.py 내 IMAGE_DIR 경로를 실제 데이터 폴더로 수정
python src/train.py
# → swin_dino_ep2.pth, ep4.pth, ... ep15.pth 생성
```

### 3. 동작 확인 (Sanity Check)

```bash
python src/sanity_check.py
# 정상 출력 예시 → "모델 최종 출력 형태: [8, 65536]"
```

### 4. Stage 2 — YOLOv8 파인튜닝

```bash
# src/train_swin_yolo.py 내 WEIGHT_PATH, data 경로를 수정 후 실행
python src/train_swin_yolo.py
# → runs/detect/train/weights/best.pt 생성
```

### 5. 추론 테스트

```bash
python src/test_model.py
# → runs/detect/predict/ 폴더에 결과 이미지 저장
```

---

## 🔧 Hyperparameters — 주요 하이퍼파라미터

### Stage 1 · DINO Pretraining

| 파라미터 | 값 | 이유 |
|---|---|---|
| `out_dim` | 8192 | DINO 투영 차원 |
| `student_temp` | 0.1 | Student 분포 스무딩 |
| `teacher_temp` | 0.07 | Teacher 분포 날카롭게 (모드 콜랩스 방지) |
| `center_momentum` | 0.9 | Center EMA 안정성 |
| `lr` | 5e-5 | 사전학습 Swin 가중치 보호 |
| `weight_decay` | 0.04 | AdamW 정규화 |
| `teacher_momentum` | 0.996 | Teacher EMA 안정성 |
| `max_norm` | 3.0 | Gradient Clipping |
| `epochs` | 15 | 사전학습 에폭 |
| `batch_size` | 16 | 듀얼 GPU 기준 |
| `global_size` | 224 | 글로벌 크롭 크기 |
| `local_size` | 224 | 로컬 크롭 출력 크기 (소형 객체 확대) |
| `local_scale` | 0.05 ~ 0.4 | 로컬 크롭 원본 스케일 범위 |

### Stage 2 · YOLOv8 Fine-tuning

| 파라미터 | 값 | 이유 |
|---|---|---|
| `imgsz` | 640 | YOLOv8 표준 입력 해상도 |
| `epochs` | 50 | 파인튜닝 에폭 |
| `batch` | 8 | GPU 메모리 기준 |
| `optimizer` | AdamW | |
| `lr0` | 1e-4 | Neck/Head 학습률 |
| `backbone freeze` | True | 사전학습 가중치 보존 |
| `nc` | 10 | 탐지 클래스 수 |

---

## 🗂️ Dataset — 데이터셋

| 항목 | 값 |
|---|---|
| Pretrain (unlabeled) | **161,381 장** |
| Finetune Train | **161,381 장** |
| Finetune Validation | **10,086 장** |
| Classes | 10 (Fishing_Boat, Merchant_Ship, Warship, Person, Bird, Fixed_Wing, Rotary_Wing, UAV, Leaflet, Trash_Bomb) |

---

## 📜 License & Acknowledgements

- **YOLOv8** — [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Swin Transformer** — Liu et al., *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"*, ICCV 2021
- **DINO** — Caron et al., *"Emerging Properties in Self-Supervised Vision Transformers"*, ICCV 2021
- **timm** — [rwightman/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

---

<p align="center">
  <sub>🛰️ KSU · Small Object Detection Project</sub>
</p>
