from ultralytics import YOLO
from datetime import datetime

model = YOLO("best.engine", task="detect")

results = model.val(
    data="/home/risenano02/works/data.yaml",
    imgsz=640,
    batch=1,
    device=0,
)

# 클래스 이름
names = results.names
nc = len(names)

# 전체 메트릭
metrics = results.results_dict
precision = metrics["metrics/precision(B)"]
recall = metrics["metrics/recall(B)"]
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
map50 = metrics["metrics/mAP50(B)"]
map50_95 = metrics["metrics/mAP50-95(B)"]

# 추론 속도 (preprocess, inference, postprocess in ms)
speed = results.speed
infer_ms = speed["inference"]
total_ms = speed["preprocess"] + speed["inference"] + speed["postprocess"]
fps = 1000.0 / total_ms if total_ms > 0 else 0.0

# 클래스별 메트릭
class_ap50 = results.box.ap50
class_ap50_95 = results.box.ap

# 결과 문자열
lines = []
lines.append("=" * 70)
lines.append(f"  YOLOv8 TensorRT Engine Benchmark Results")
lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 70)
lines.append("")
lines.append("[Overall Metrics]")
lines.append(f"  Precision:       {precision:.4f}")
lines.append(f"  Recall:          {recall:.4f}")
lines.append(f"  F1 Score:        {f1:.4f}")
lines.append(f"  mAP@50:          {map50:.4f}")
lines.append(f"  mAP@50-95:       {map50_95:.4f}")
lines.append("")
lines.append("[Inference Speed]")
lines.append(f"  Preprocess:      {speed['preprocess']:.2f} ms")
lines.append(f"  Inference:       {infer_ms:.2f} ms")
lines.append(f"  Postprocess:     {speed['postprocess']:.2f} ms")
lines.append(f"  Total:           {total_ms:.2f} ms")
lines.append(f"  FPS:             {fps:.2f}")
lines.append("")
lines.append("[Per-Class Metrics]")
lines.append(f"  {'Class':<20s} {'mAP@50':>10s} {'mAP@50-95':>12s}")
lines.append(f"  {'-'*20} {'-'*10} {'-'*12}")
for i in range(nc):
    cname = names[i]
    ap50 = class_ap50[i]
    ap = class_ap50_95[i]
    lines.append(f"  {cname:<20s} {ap50:>10.4f} {ap:>12.4f}")
lines.append("=" * 70)

output = "\n".join(lines)
print(output)

with open("benchmark_result.txt", "w") as f:
    f.write(output + "\n")

print(f"\nResults saved to benchmark_result.txt")
