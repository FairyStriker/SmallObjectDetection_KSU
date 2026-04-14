# 📊 Benchmark — Jetson Orin Nano (TensorRT FP16)

> **YOLOv8 + Swin-DINO 백본** 모델을 **NVIDIA Jetson Orin Nano (Super) 8GB**에서
> **TensorRT FP16 엔진**으로 변환 후 측정한 벤치마크 결과입니다.

---

## 🖥️ Benchmark Environment — 벤치마크 환경

| 항목 | 값 |
|------|-----|
| **Device** | NVIDIA Jetson Orin Nano Dev Kit (Super) — 8GB RAM |
| **JetPack** | 6.2.1 |
| **L4T** | 36.4.7 (R36 Rev.4.7, 2025-09-18) |
| **Power Mode** | `MAXN_SUPER` (mode 2) |
| **CUDA** | 12.6 (V12.6.68) |
| **cuDNN** | 9.3.0.75 |
| **TensorRT** | 10.3.0.30 |
| **Python** | 3.10.12 |
| **PyTorch** | 2.5.0 (NVIDIA Jetson build `nv24.08`, CUDA 빌드) |
| **torchvision** | 0.20.0 |
| **Ultralytics** | 8.4.24 |
| **Precision** | FP16 |
| **Input Size** | 640 × 640 |
| **Batch Size** | 1 |

---

## 📈 Overall Metrics — 전체 성능

| Metric | Value |
|--------|-------|
| **Precision** | **0.9817** |
| **Recall** | **0.9764** |
| **F1 Score** | **0.9790** |
| **mAP@0.5** | **0.9861** |
| **mAP@0.5–0.95** | **0.8673** |

---

## ⚡ Inference Speed — 추론 속도 (Jetson Orin Nano, TensorRT FP16)

| Stage | Latency |
|-------|---------|
| Preprocess | 1.96 ms |
| **Inference** | **116.89 ms** |
| Postprocess | 4.34 ms |
| **Total** | **123.19 ms** |
| **FPS** | **8.12** |

> 측정 조건: `imgsz=640`, `batch=1`, Jetson Orin Nano (Super) `MAXN_SUPER` 전원 모드

---

## 🎯 Per-class Performance — 클래스별 성능

| Class | mAP@0.5 | mAP@0.5–0.95 |
|-------|--------:|-------------:|
| Fishing_Boat   | 0.9941 | 0.8773 |
| Merchant_Ship  | 0.9875 | 0.9068 |
| Warship        | 0.9944 | 0.9258 |
| Person         | 0.9319 | 0.5765 |
| Bird           | 0.9781 | 0.7603 |
| Fixed_Wing     | 0.9950 | 0.9377 |
| Rotary_Wing    | 0.9950 | 0.9564 |
| UAV            | 0.9950 | 0.8891 |
| Leaflet        | 0.9950 | 0.8832 |
| Trash_Bomb     | 0.9950 | 0.9593 |
| **All (mean)** | **0.9861** | **0.8673** |

> 💡 `Person` 클래스의 mAP@0.5–0.95가 낮은 것은 작은 해상도/다양한 포즈/부분 가림 등의 요인으로 해석됩니다.

---

## 📉 Curves — 성능 곡선

### Precision-Recall Curve
![PR Curve](curves/PR_curve.png)

### F1-Confidence Curve
![F1 Curve](curves/F1_curve.png)

### Precision-Confidence Curve
![Precision Curve](curves/P_curve.png)

### Recall-Confidence Curve
![Recall Curve](curves/R_curve.png)

---

## 🔲 Confusion Matrix

| Raw | Normalized |
|-----|------------|
| ![CM](curves/confusion_matrix.png) | ![CM Normalized](curves/confusion_matrix_normalized.png) |

---

## 🖼️ Sample Predictions — 샘플 예측 결과

| Ground Truth | Prediction |
|--------------|------------|
| ![GT](samples/val_batch0_labels.jpg) | ![Pred](samples/val_batch0_pred.jpg) |

---

## 🔁 How to Reproduce — 재현 방법

Jetson Orin Nano 환경에서 다음 순서로 실행하면 동일한 결과를 얻을 수 있습니다.

### 1. PyTorch 모델 → TensorRT FP16 엔진 변환

```bash
python src/export_tensorrt.py \
    --weights runs/detect/train/weights/best.pt \
    --imgsz 640 \
    --workspace 4
# → best.engine 생성
```

### 2. 벤치마크 실행

```bash
# 전원 모드를 MAXN_SUPER로 설정
sudo nvpmodel -m 2
sudo jetson_clocks

# benchmark_engine.py 내의 data.yaml 경로를 수정 후 실행
python src/benchmark_engine.py
# → benchmark_result.txt 생성
```

> ⚠️ TensorRT 엔진 파일은 **디바이스 종속적**이므로 반드시 **Jetson Orin Nano 위에서 변환**해야 하며,
> PC에서 생성한 엔진은 Jetson에서 동작하지 않습니다.

---

## 📄 Raw Output

전체 원본 결과는 [`benchmark_result.txt`](benchmark_result.txt)에서 확인할 수 있습니다.
