import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from data_aug import DINOSmallObjectAugmentation
from model import MultiCropSwinDINO

# --- 1. DINO 전용 Loss 함수 정의 ---
class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(8) 

        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2) 

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

# --- 2. Teacher 모델 EMA 업데이트 함수 ---
@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
        param_teacher.data.mul_(momentum).add_((1 - momentum) * param_student.data)

# --- 3. 메인 학습 루프 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    # [최적화 1] 8GB VRAM에 맞게 출력 차원 축소
    OUT_DIM = 8192 

    transform = DINOSmallObjectAugmentation(global_size=224, local_size=224, local_crops_number=6)
    dataset = datasets.FakeData(size=32, image_size=(3, 500, 500), transform=transform)
    
    # [최적화 2] 메모리 확보를 위해 배치 사이즈 축소 (4 -> 2)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    student = MultiCropSwinDINO(out_dim=OUT_DIM).to(device)
    teacher = MultiCropSwinDINO(out_dim=OUT_DIM).to(device)
    
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False

    dino_loss = DINOLoss(out_dim=OUT_DIM).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=0.0005, weight_decay=0.04)

    # [최적화 3] AMP (Mixed Precision) 스케일러 초기화
    scaler = torch.cuda.amp.GradScaler()

    epochs = 2 
    print("🚀 학습 테스트를 시작합니다...")

    for epoch in range(epochs):
        for it, (images, _) in enumerate(data_loader):
            images = [im.to(device) for im in images]
            
            # AMP 적용 구역: 메모리 사용량 절반 감소, 연산 속도 증가
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    teacher_output = teacher(images[:2])
                
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output)
            
            optimizer.zero_grad()
            # Scaler를 이용한 역전파 및 최적화
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            update_teacher(student, teacher, momentum=0.996)
            
            print(f"Epoch [{epoch+1}/{epochs}] | Step [{it+1}/{len(data_loader)}] | Loss: {loss.item():.4f}")

    print("✅ 학습 루프 테스트가 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()