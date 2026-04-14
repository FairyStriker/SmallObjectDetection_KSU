import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PIL import Image
from data_aug import DINOSmallObjectAugmentation
from model import MultiCropSwinDINO

# 1. 가짜 데이터(Dummy Image) 생성
dummy_image = Image.new('RGB', (500, 500), color='red')

# 2. 증강 파이프라인 통과 (수정된 부분: local_size를 224로 통일하여 Zoom-in 효과 부여)
transform = DINOSmallObjectAugmentation(global_size=224, local_size=224, local_crops_number=6)
crops = transform(dummy_image)

print(f"생성된 크롭 개수: {len(crops)}")
print(f"Global Crop 1 형태: {crops[0].shape}")
print(f"Local Crop 1 형태: {crops[2].shape}") # 이제 이것도 [3, 224, 224]로 출력됩니다.

# 3. 모델 통과 테스트
model = MultiCropSwinDINO()
# 배치 차원(B) 추가 (1장의 이미지 크롭들을 배치처럼 묶어줌)
crops_batched = [crop.unsqueeze(0) for crop in crops] 

output = model(crops_batched)
print(f"모델 최종 출력 형태: {output.shape}") 
# 정상 작동 시: [8, 65536] (크롭 8개 x DINO Head 출력 차원)