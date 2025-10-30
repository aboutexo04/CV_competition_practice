import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import copy
from pathlib import Path
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# config에서 설정값 가져오기 (옵션)
try:
    from .config import SELECTED_MODEL, IMAGE_SIZE
except ImportError:
    # config를 못찾으면 기본값 사용
    SELECTED_MODEL = 'efficientnet_b0'
    IMAGE_SIZE = 224

# 디바이스 설정
# 🔥 ConvNeXt와 일부 모델은 MPS에서 view() 문제가 있어 CPU 사용
PROBLEMATIC_MODELS = ['convnext_tiny', 'convnext_small', 'convnext_base']

if SELECTED_MODEL in PROBLEMATIC_MODELS:
    device = torch.device("cpu")
    print(f'Using device: CPU (MPS has compatibility issues with {SELECTED_MODEL})')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f'Using device: MPS (Apple Silicon)')
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'Using device: CUDA')
else:
    device = torch.device("cpu")
    print(f'Using device: CPU')

# ============================================
# Train Transform - 100% 작동 보장! ⭐⭐⭐⭐⭐
# ============================================
train_transform = A.Compose([
    # 필수: 리사이즈
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    
    # 문서 특화 Augmentation
    A.Rotate(limit=5, p=0.3),  # 살짝 회전
    A.Perspective(scale=0.05, p=0.3),  # 각도 변화
    
    # 노이즈 추가 (스캔 효과)
    A.GaussNoise(p=0.2),
    
    # 흐림 효과 (압축/스캔 효과)
    A.Blur(blur_limit=3, p=0.2),
    
    # 밝기/대비 조정 (매우 중요!)
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    
    # 선택: 추가 Augmentation
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=5,
        p=0.3
    ),
    
    # Normalize (필수!)
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    
    ToTensorV2()
])

# ============================================
# Validation Transform
# ============================================
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# ============================================
# CIFAR10 데이터 직접 로딩 (pickle 파일에서)
# ============================================


class CIFAR10Dataset(Dataset):
    """
    CIFAR10 데이터셋을 pickle 파일에서 직접 로드
    
    Args:
        data_dir: CIFAR10 데이터가 있는 디렉토리 (cifar-10-batches-py 폴더)
        train: True면 train 데이터, False면 test 데이터
        transform: Albumentations transform
        indices: 사용할 인덱스 리스트 (서브샘플링용, None이면 전체)
    """
    def __init__(self, data_dir, train=True, transform=None, indices=None):
        self.data_dir = Path(data_dir)
        self.train = train
        self.transform = transform
        
        # CIFAR10 데이터 로드
        self.data = []
        self.labels = []
        
        if train:
            # Train 데이터: data_batch_1 ~ data_batch_5
            for i in range(1, 6):
                batch_path = self.data_dir / f'data_batch_{i}'
                with open(batch_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    # CIFAR10은 항상 bytes 키를 사용
                    labels = batch.get(b'labels', batch.get('labels', []))
                    batch_data = batch.get(b'data', batch.get('data', None))
                    if batch_data is not None:
                        self.data.append(batch_data)
                        self.labels.extend(labels)
                    else:
                        raise ValueError(f"CIFAR10 data_batch_{i} file is missing 'data' key")
            
            self.data = np.vstack(self.data)  # (50000, 3072)
        else:
            # Test 데이터: test_batch
            batch_path = self.data_dir / 'test_batch'
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                # CIFAR10은 항상 bytes 키를 사용
                # labels 처리
                labels = batch.get(b'labels', batch.get('labels', []))
                
                # data 처리
                batch_data = batch.get(b'data', batch.get('data', None))
                if batch_data is not None:
                    self.data = batch_data
                    self.labels = labels
                else:
                    raise ValueError(f"CIFAR10 test batch file is missing 'data' key")
        
        # 데이터 형태 변환 (3072 -> 32x32x3)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
        
        # 특정 인덱스만 사용
        if indices is not None:
            self.data = self.data[indices]
            self.labels = [self.labels[i] for i in indices]
            self.indices = indices
        else:
            self.indices = list(range(len(self.data)))
        
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 이미지 가져오기 (이미 numpy array)
        image = self.data[idx]  # (32, 32, 3)
        
        # 레이블 가져오기
        label = int(self.labels[idx])
        
        # Transform 적용
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
# ============================================
# CIFAR10 데이터 로딩 (pickle 파일에서 직접 로드)
# ============================================

print("="*60)
print("📦 Loading CIFAR10 Data (from pickle files)")
print("="*60)

# 데이터 경로 설정 (프로젝트 루트 기준)
# 현재 파일(data.py)의 위치에서 프로젝트 루트를 찾아서 절대 경로로 변환
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
data_dir = project_root / 'data' / 'cifar-10-batches-py'

# CIFAR10 데이터 직접 로드
print("✅ Loading CIFAR10 data from pickle files...")
train_data = CIFAR10Dataset(
    data_dir=str(data_dir),
    train=True,
    transform=None  # 나중에 transform 적용
)

test_data = CIFAR10Dataset(
    data_dir=str(data_dir),
    train=False,
    transform=None
)

# 클래스 이름 로드
meta_path = data_dir / 'batches.meta'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f, encoding='bytes')
    if b'label_names' in meta:
        class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    else:
        class_names = meta.get('label_names', [f'class_{i}' for i in range(10)])

num_classes = len(class_names)

print(f"\n✅ CIFAR10 Data Loaded!")
print(f"Train: {len(train_data):,} images")
print(f"Test:  {len(test_data):,} images")
print(f"Classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Location: {data_dir.absolute()}")

import random

# 🔥 연습용 설정: 데이터셋 크기 조절 (10% 사용)
USE_SUBSET = True  # False로 변경하면 전체 데이터 사용
SUBSET_RATIO = 0.1  # 10%만 사용 (0.1 ~ 1.0)

# train_data와 test_data는 이미 CIFAR10Dataset으로 로드됨
# 레이블 추출 (K-Fold용)
train_labels = train_data.labels.tolist()

# 🔥 데이터셋 서브샘플링 (클래스별 균등 샘플링)
if USE_SUBSET:
    # Train 데이터 서브샘플링
    train_indices_by_class = {}
    for idx, label in enumerate(train_labels):
        if label not in train_indices_by_class:
            train_indices_by_class[label] = []
        train_indices_by_class[label].append(idx)
    
    # 각 클래스에서 균등하게 샘플링
    selected_train_indices = []
    for label, indices in train_indices_by_class.items():
        n_samples = int(len(indices) * SUBSET_RATIO)
        selected_train_indices.extend(random.sample(indices, n_samples))
    
    selected_train_indices.sort()
    train_dataset_raw = CIFAR10Dataset(
        data_dir=train_data.data_dir,
        train=True,
        transform=None,
        indices=selected_train_indices
    )
    train_labels = [train_labels[i] for i in selected_train_indices]
    
    # Test 데이터 서브샘플링
    test_labels = test_data.labels.tolist()
    test_indices_by_class = {}
    for idx, label in enumerate(test_labels):
        if label not in test_indices_by_class:
            test_indices_by_class[label] = []
        test_indices_by_class[label].append(idx)
    
    selected_test_indices = []
    for label, indices in test_indices_by_class.items():
        n_samples = int(len(indices) * SUBSET_RATIO)
        selected_test_indices.extend(random.sample(indices, n_samples))
    
    selected_test_indices.sort()
    test_dataset_raw = CIFAR10Dataset(
        data_dir=test_data.data_dir,
        train=False,
        transform=None,  # 나중에 K-Fold에서 적용
        indices=selected_test_indices
    )
    
    print(f"🔥 발열 감소 모드: 데이터셋 {int(SUBSET_RATIO*100)}% 사용")
else:
    train_dataset_raw = train_data
    test_dataset_raw = test_data

# Transform 적용된 test dataset
# USE_SUBSET일 때는 서브샘플링된 것 사용, 아니면 전체 사용
if USE_SUBSET:
    # 서브샘플링된 test_dataset_raw를 transform과 함께 다시 생성
    # test_dataset_raw가 이미 indices로 생성되어 있으므로, transform만 적용
    if 'val_transform' in globals() and val_transform is not None:
        test_dataset = CIFAR10Dataset(
            data_dir=test_data.data_dir,
            train=False,
            transform=val_transform,
            indices=selected_test_indices if USE_SUBSET else None
        )
    else:
        # val_transform이 없으면 나중에 적용할 수 있도록 저장
        test_dataset = test_dataset_raw  # transform은 None이지만 나중에 적용
else:
    # 전체 데이터 사용
    if 'val_transform' in globals() and val_transform is not None:
        test_dataset = CIFAR10Dataset(
            data_dir=test_data.data_dir,
            train=False,
            transform=val_transform
        )
    else:
        test_dataset = CIFAR10Dataset(
            data_dir=test_data.data_dir,
            train=False,
            transform=None
        )

if USE_SUBSET:
    print(f'✅ Total train size: {len(train_dataset_raw):,}')
    print(f'✅ Test size: {len(test_dataset_raw):,}')
else:
    print(f'✅ Total train size: {len(train_data):,}')
    print(f'✅ Test size: {len(test_dataset):,}')

if USE_SUBSET:
    print(f"\n💡 전체 데이터로 학습하려면 USE_SUBSET = False로 변경하세요")