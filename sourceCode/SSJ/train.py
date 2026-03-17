import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from data_aug import DINOSmallObjectAugmentation
from model import MultiCropSwinDINO

# --- 1. 커스텀 데이터셋 ---
class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        return image

# --- 2. DINO Loss ---
class DINOLoss(nn.Module):
    # [수정 포인트 1] teacher_temp 기본값을 0.07로 올려서 모델의 꼼수 차단
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.07, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(8, dim=0) 

        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2, dim=0)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: continue 
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# --- 3. Teacher 업데이트 (멀티 GPU 대응) ---
@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    student_params = student.module.parameters() if isinstance(student, nn.DataParallel) else student.parameters()
    teacher_params = teacher.module.parameters() if isinstance(teacher, nn.DataParallel) else teacher.parameters()
    
    for param_student, param_teacher in zip(student_params, teacher_params):
        param_teacher.data.mul_(momentum).add_((1 - momentum) * param_student.data)

# --- 4. 메인 학습 루프 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 사용 중인 디바이스: {device} | GPU 개수: {torch.cuda.device_count()}개")

    OUT_DIM = 8192 
    transform = DINOSmallObjectAugmentation(global_size=224, local_size=224, local_crops_number=6)
    
    # 🚨 여기에 우분투 환경의 16만 장 이미지 폴더 경로를 입력하세요! (예: /home/user/DATA/images)
    IMAGE_DIR = "여기에_폴더_경로를_적어주세요" 
    
    dataset = UnlabeledImageDataset(image_dir=IMAGE_DIR, transform=transform)
    print(f"총 {len(dataset)}장의 실제 이미지를 불러왔습니다.")
    
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    student = MultiCropSwinDINO(out_dim=OUT_DIM).to(device)
    teacher = MultiCropSwinDINO(out_dim=OUT_DIM).to(device)
    
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False

    if torch.cuda.device_count() > 1:
        print("🚀 듀얼 GPU(DataParallel) 모드를 활성화합니다!")
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)

    dino_loss = DINOLoss(out_dim=OUT_DIM, teacher_temp=0.07).to(device)
    
    # [수정 포인트 2] 학습률을 0.00005 로 대폭 낮춰서 기존 백본의 뇌가 망가지는 것을 방지
    optimizer = optim.AdamW(student.parameters(), lr=0.00005, weight_decay=0.04)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 15 
    print("🔥 모드 콜랩스 방지 코드 적용 완료! 16만 장 듀얼 GPU 집중 학습을 시작합니다...")

    for epoch in range(epochs):
        for it, images in enumerate(data_loader): 
            teacher_inputs = torch.cat(images[:2], dim=0).to(device, non_blocking=True)
            student_inputs = torch.cat(images, dim=0).to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    teacher_output = teacher(teacher_inputs)
                student_output = student(student_inputs)
            
            # Loss 계산 시 무조건 32비트(float)로 변환하여 언더플로우 방지
            loss = dino_loss(student_output.float(), teacher_output.float())
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 기울기 폭발(Gradient Spike) 방지
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            update_teacher(student, teacher, momentum=0.996)
            
            # 10 스텝마다 로그를 출력합니다.
            if (it + 1) % 10 == 0: 
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{it+1}/{len(data_loader)}] | Loss: {loss.item():.4f}")
        
        # 2 에폭마다 가중치 저장
        if (epoch + 1) % 2 == 0:
            save_path = f"swin_dino_ep{epoch+1}.pth"
            torch.save(student.module.backbone.state_dict(), save_path)
            print(f"💾 모델 저장 완료: {save_path}")

if __name__ == '__main__':
    main()
