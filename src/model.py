import torch  # <-- 이 줄이 추가되었습니다!
import torch.nn as nn
import timm

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048):
        super().__init__()
        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # L2 Normalization 후 적용될 Weight Normalized Linear Layer
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropSwinDINO(nn.Module):
    def __init__(self, out_dim=65536):
        super().__init__()
        # Swin-Tiny 백본 로드 (timm 사용, 분류기 제거)
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features
        self.head = DINOHead(in_dim=embed_dim, out_dim=out_dim)

    def forward(self, x):
        # Multi-crop 리스트가 들어왔을 때 처리
        if isinstance(x, list):
            # 모든 크롭의 텐서 리스트를 받아서 처리
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]), 
                return_counts=True, 
            )[1], 0)
            
            # 해상도가 같은 크롭끼리 배치로 묶어서 백본 통과
            start_idx = 0
            output = []
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx:end_idx]))
                output.append(_out)
                start_idx = end_idx
            
            # Head 통과
            output = self.head(torch.cat(output))
            return output
        else:
            # 단일 텐서일 경우
            return self.head(self.backbone(x))