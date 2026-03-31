# Swin-DINO-YOLOv8 소형 객체 탐지

> **Swin Transformer** 백본을 **DINO 자기지도학습**으로 사전학습한 뒤, **YOLOv8** 탐지 프레임워크에 이식해 10개 클래스 소형 객체를 탐지하는 2단계 학습 파이프라인입니다.

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [아키텍처 설계 의도](#아키텍처-설계-의도)
- [전체 파이프라인](#전체-파이프라인)
- [1단계 — DINO 자기지도 사전학습](#1단계--dino-자기지도-사전학습)
- [2단계 — YOLOv8 파인튜닝](#2단계--yolov8-파인튜닝)
- [백본 교체의 의미](#백본-교체의-의미)
- [파일 구조](#파일-구조)
- [실행 방법](#실행-방법)
- [주요 하이퍼파라미터](#주요-하이퍼파라미터)

---

## 프로젝트 개요

기존 YOLOv8은 **CSPDarknet** (CNN 기반 백본)을 사용합니다. CNN은 고정된 커널 크기로 주변 픽셀만 참조하기 때문에, 이미지 전체의 문맥을 파악하는 데 한계가 있습니다.

이 프로젝트는 두 가지 개선을 결합합니다.

1. **Swin Transformer 백본**: Shifted Window Attention으로 이미지 전체의 전역 관계를 파악
2. **DINO 사전학습**: 라벨 없는 이미지 16만 장으로 소형 객체에 민감한 특징 추출 능력을 선학습

결과적으로 "YOLO의 빠른 탐지 프레임워크"에 "Transformer의 전역 문맥 이해 능력"을 결합한 구조입니다.

---

## 아키텍처 설계 의도

### 기존 YOLOv8

```
입력 이미지 → CSPDarknet (CNN 백본) → FPN Neck → Detect Head → 탐지 결과
```

### 본 프로젝트

```
입력 이미지 → Swin Transformer 백본 (DINO 사전학습) → FPN Neck → Detect Head → 탐지 결과
```

### CSPDarknet vs Swin Transformer 비교

| 항목 | CSPDarknet (기존) | Swin Transformer (본 프로젝트) |
|---|---|---|
| 연산 방식 | 로컬 컨볼루션 | Shifted Window Attention |
| 수용 영역 | 레이어가 깊어질수록 점진적 확장 | 전역 문맥을 직접 파악 |
| 사전학습 | ImageNet 분류 | DINO 자기지도학습 (16만 장) |
| 소형 객체 | 상대적으로 취약 | 로컬 크롭 학습으로 강화 |

---

## 전체 파이프라인

```
[라벨 없는 이미지 16만 장]
        ↓
┌─────────────────────────────────────────────────────┐
│  1단계: DINO 자기지도 사전학습 (train.py)            │
│                                                     │
│  멀티크롭 증강 → Student (Swin + DINOHead)          │
│                → Teacher (Swin + DINOHead, EMA)     │
│                → DINO Loss                         │
│                                                     │
│  ※ 15 에폭 완료 후 Swin 백본 가중치만 저장          │
│     (DINOHead는 역할 완료 후 폐기)                  │
└─────────────────────────────────────────────────────┘
        ↓  swin_dino_ep15.pth (백본 가중치만)
┌─────────────────────────────────────────────────────┐
│  2단계: YOLOv8 파인튜닝 (train_swin_yolo.py)        │
│                                                     │
│  Swin 백본 이식 (동결) → FPN Neck → Detect Head     │
│  ※ Neck + Head만 학습, 백본은 건드리지 않음         │
└─────────────────────────────────────────────────────┘
        ↓
[10개 클래스 소형 객체 탐지 모델 완성]
```

---

## 1단계 — DINO 자기지도 사전학습

### 핵심 아이디어

DINOHead는 Swin 백본이 좋은 특징을 뽑도록 **학습 신호를 주는 임시 도구**입니다.
학습이 완료되면 DINOHead는 폐기하고, 백본 가중치만 저장합니다.

> 비유하자면, DINOHead는 Swin을 가르치는 **임시 선생님**이고 졸업 후에는 필요 없어집니다.
> 중요한 것은 Swin 백본에 녹아든 **"잘 보는 능력"** 입니다.

### 모델 구조 (`model.py`)

```python
class MultiCropSwinDINO(nn.Module):
    def __init__(self, out_dim=65536):
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0
        )  # Swin 백본
        self.head = DINOHead(in_dim=embed_dim, out_dim=out_dim)  # 임시 투영 헤드
```

`DINOHead`는 3층 MLP → L2 정규화 → Weight-Norm Linear 구조로 특징을 8192차원으로 압축합니다.

### 멀티크롭 증강 (`data_aug.py`)

한 장의 이미지에서 8개의 크롭을 동시에 생성합니다.

| 크롭 종류 | 개수 | 크기 | 스케일 범위 | 목적 |
|---|---|---|---|---|
| Global View | 2장 | 224×224 | 0.4 ~ 1.0 | 전체 맥락 파악 |
| Local View | 6장 | 224×224 | **0.05 ~ 0.4** | **소형 객체 확대 학습** |

`local_size=224`로 작은 영역을 확대해서 보는 것이 소형 객체 탐지 성능을 높이기 위한 핵심 설계 선택입니다.

### Student-Teacher 학습 구조

```
이미지 8크롭 → Student (전체 8장 처리) ─┐
이미지 2크롭 → Teacher (글로벌 2장만) ─┤→ DINO Loss → Student만 역전파
                   ↑                    │
                EMA 업데이트 ←──────────┘
               (momentum=0.996)
```

- **Student**: 역전파로 직접 학습
- **Teacher**: Student의 EMA(지수이동평균)로만 업데이트 (gradient 없음)
- Teacher가 더 안정적인 표현을 유지하기 때문에, Student가 Teacher를 모방하는 과정에서 고품질 특징을 학습합니다.

### 모드 콜랩스 방지 (`train.py`)

DINO 학습에서 모든 입력에 같은 출력을 내는 **모드 콜랩스**가 가장 큰 위험입니다.
세 가지 메커니즘으로 방지합니다.

```python
# 1. Centering: Teacher 출력의 평균을 빼서 특정 차원 지배를 억제
teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)

# 2. Sharpening: teacher_temp=0.07로 분포를 날카롭게 유지
# 3. Gradient Clipping: 기울기 폭발 방지
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
```

추가로 `lr=0.00005`의 극도로 낮은 학습률을 사용해 사전학습된 Swin 가중치가 파괴되지 않도록 합니다.

### 듀얼 GPU 지원

```python
if torch.cuda.device_count() > 1:
    student = nn.DataParallel(student)
    teacher = nn.DataParallel(teacher)

# Teacher EMA 업데이트 시 DataParallel 래핑 자동 처리
student_params = student.module.parameters() if isinstance(student, nn.DataParallel) \
                 else student.parameters()
```

### 가중치 저장

```python
# DINOHead는 버리고 Swin 백본 가중치만 저장
torch.save(student.module.backbone.state_dict(), f"swin_dino_ep{epoch+1}.pth")
```

---

## 2단계 — YOLOv8 파인튜닝

### 핵심 아이디어: 백본 이식 수술

`yolov8_swin.yaml`의 backbone 섹션은 처음부터 **자리만 잡아주는 더미 레이어**입니다.
`inject_swin()`이 이 자리를 Swin Transformer로 물리적으로 교체합니다.

### YAML 더미 구조 (`yolov8_swin.yaml`)

```yaml
backbone:
  - [-1, 1, Conv, [192, 3, 2]]  # 0: 더미 → SwinStage0로 교체 (P3, 192ch)
  - [-1, 1, Conv, [384, 3, 2]]  # 1: 더미 → SwinStage1로 교체 (P4, 384ch)
  - [-1, 1, Conv, [768, 3, 2]]  # 2: 더미 → SwinStage2로 교체 (P5, 768ch)
```

### 백본 이식 구조 (`train_swin_yolo.py`)

Swin은 하나의 forward pass에서 P3/P4/P5를 동시에 계산합니다.
YOLOv8은 레이어를 순차적으로 호출하므로, 이를 맞추기 위해 Adapter 패턴을 사용합니다.

```
SwinStage0: Swin 실제 연산 수행
            ├── P3 (192ch) → 즉시 출력 (YOLOv8 레이어 0의 출력)
            ├── P4 (384ch) → self.p4에 저장
            └── P5 (768ch) → self.p5에 저장

SwinStage1: self.stage0.p4 반환 (YOLOv8 레이어 1의 출력)
SwinStage2: self.stage0.p5 반환 (YOLOv8 레이어 2의 출력)
```

```python
def inject_swin(model, weight_path):
    s0 = SwinStage0(weight_path)  # 실제 Swin 연산 + 가중치 로드
    s1 = SwinStage1(s0)           # P4 adapter
    s2 = SwinStage2(s0)           # P5 adapter

    model.model[0] = s0  # 더미 Conv 교체
    model.model[1] = s1
    model.model[2] = s2
```

### Neck & Head 구조 (`yolov8_swin.yaml`)

```yaml
head:
  # P5 → P4 업샘플링 후 Concat
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 1], 1, Concat, [1]]
  - [-1, 3, C2f, [384]]

  # P4 → P3 업샘플링 후 Concat
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 0], 1, Concat, [1]]
  - [-1, 3, C2f, [192]]  # P3 탐지 헤드 (소형 객체)

  # P3 → P4 다운샘플링
  - [-1, 1, Conv, [192, 3, 2]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 3, C2f, [384]]  # P4 탐지 헤드 (중형 객체)

  # P4 → P5 다운샘플링
  - [-1, 1, Conv, [384, 3, 2]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 3, C2f, [768]]  # P5 탐지 헤드 (대형 객체)

  - [[8, 11, 14], 1, Detect, [nc]]  # 최종 탐지 (P3, P4, P5 멀티스케일)
```

### 학습 설정

```python
model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=224,       # 사전학습과 동일한 해상도 유지
    batch=16,
    device=0,
    optimizer='AdamW',
    lr0=0.001        # Neck/Head 학습률 (백본보다 높게)
)
```

Swin 백본은 `requires_grad=False`로 동결되어 **Neck과 Detect Head만 학습**됩니다.

---

## 백본 교체의 의미

### 특징맵 채널 수 비교

| 특징맵 | 기존 CSPDarknet | Swin Transformer |
|---|---|---|
| P3 (소형 객체) | 256ch | **192ch** |
| P4 (중형 객체) | 512ch | **384ch** |
| P5 (대형 객체) | 1024ch | **768ch** |

Swin의 계층적 구조 덕분에 P3/P4/P5가 자연스럽게 생성되며, YOLO의 FPN Neck과 채널 수를 맞추도록 YAML을 설계했습니다.

### 왜 DINO 사전학습이 탐지에 유리한가

DINO는 분류 레이블 없이 **이미지 내 구조적 패턴**을 스스로 학습합니다.
로컬 크롭(scale 0.05~0.4)으로 작은 영역을 확대 학습하기 때문에,
일반적인 ImageNet 분류 사전학습보다 소형 객체의 경계와 텍스처에 더 민감한 특징을 형성합니다.

---

## 파일 구조

```
.
├── train.py               # 1단계: DINO 자기지도 사전학습
├── model.py               # MultiCropSwinDINO (Swin + DINOHead)
├── data_aug.py            # 멀티크롭 증강 (글로벌 2 + 로컬 6)
├── train_swin_yolo.py     # 2단계: YOLOv8 파인튜닝 + 백본 이식
├── yolov8_swin.yaml       # YOLOv8 커스텀 구조 (더미 백본 + Neck/Head)
├── sanity_check.py        # 모델 동작 확인용
├── test_model.py          # 학습 완료 모델 추론 테스트
└── dataset/
    └── data.yaml          # 데이터셋 경로 및 클래스 설정
```

---

## 실행 방법

### 환경 설치

```bash
pip install torch torchvision timm ultralytics
```

### 1단계: DINO 사전학습

```bash
# train.py 내 IMAGE_DIR 경로 설정 후 실행
python train.py
# → swin_dino_ep2.pth, ep4.pth, ... ep15.pth 생성
```

### 동작 확인

```bash
python sanity_check.py
# 정상 출력: 모델 최종 출력 형태: [8, 65536]
```

### 2단계: YOLOv8 파인튜닝

```bash
# train_swin_yolo.py 내 WEIGHT_PATH, data 경로 설정 후 실행
python train_swin_yolo.py
# → runs/detect/train/weights/best.pt 생성
```

### 추론 테스트

```bash
python test_model.py
# → runs/detect/predict/ 폴더에 결과 이미지 저장
```

---

## 주요 하이퍼파라미터

### 1단계 (DINO 사전학습)

| 파라미터 | 값 | 이유 |
|---|---|---|
| `out_dim` | 8192 | DINO 투영 차원 |
| `student_temp` | 0.1 | Student 분포 스무딩 |
| `teacher_temp` | 0.07 | Teacher 분포 날카롭게 (모드 콜랩스 방지) |
| `center_momentum` | 0.9 | Center EMA 안정성 |
| `lr` | 0.00005 | 사전학습 Swin 가중치 보호 |
| `weight_decay` | 0.04 | AdamW 정규화 |
| `teacher momentum` | 0.996 | Teacher EMA 안정성 |
| `max_norm` | 3.0 | Gradient Clipping |
| `epochs` | 15 | 사전학습 에폭 |
| `batch_size` | 16 | 듀얼 GPU 기준 |
| `global_size` | 224 | 글로벌 크롭 크기 |
| `local_size` | 224 | 로컬 크롭 출력 크기 (소형 객체 확대) |
| `local_scale` | 0.05 ~ 0.4 | 로컬 크롭 원본 스케일 범위 |

### 2단계 (YOLOv8 파인튜닝)

| 파라미터 | 값 | 이유 |
|---|---|---|
| `imgsz` | 224 | 사전학습과 동일한 해상도 |
| `epochs` | 50 | 파인튜닝 에폭 |
| `lr0` | 0.001 | Neck/Head 학습률 |
| `optimizer` | AdamW | |
| `backbone freeze` | True | 사전학습 가중치 보존 |
| `nc` | 10 | 탐지 클래스 수 |
