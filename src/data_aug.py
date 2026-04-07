import torch
import torchvision.transforms as transforms
from PIL import Image

class DINOSmallObjectAugmentation:
    def __init__(self, global_size=224, local_size=96, local_crops_number=6):
        # 공통 정규화 설정 (ImageNet 기준)
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 1. Global View
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1), 
            transforms.RandomGrayscale(p=0.2),
            # 수정된 부분: kernel_size와 sigma 명시
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.1),
            normalize,
        ])

        # 2. Local View
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            # 수정된 부분: kernel_size와 sigma 명시
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])
        
        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = [self.global_transform(image) for _ in range(2)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops