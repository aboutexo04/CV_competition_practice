# src/data.py

import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
import pickle
import random

# ============================================
# Augmentation 함수들
# ============================================

def get_albumentations_train(img_size=224):
    """일반 이미지용 augmentation (CIFAR-10 등)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Geometric
        A.Rotate(limit=5, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            p=0.3
        ),
        A.Perspective(scale=0.05, p=0.3),
        
        # Color & Brightness
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Noise & Blur
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        
        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


try:
    from augraphy import (
        InkBleed, PaperFactory, DirtyDrum,
        Jpeg, Brightness, AugraphyPipeline
    )
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False


def get_augraphy_train(img_size=224):
    """문서 특화 augmentation (Augraphy)"""
    if not AUGRAPHY_AVAILABLE:
        print("⚠️  Augraphy not installed. Falling back to Albumentations.")
        return get_albumentations_train(img_size)
    
    # Augraphy 파이프라인 (이제 빨간 줄 안 나옴!)
    ink_phase = [
        InkBleed(intensity_range=(0.1, 0.3), p=0.3),
    ]
    
    paper_phase = [
        PaperFactory(p=0.3),
        DirtyDrum(p=0.2),
    ]
    
    post_phase = [
        Jpeg(quality_range=(60, 95), p=0.3),
        Brightness(brightness_range=(0.95, 1.05), p=0.3),
    ]
    
    augraphy_pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)
    
    return A.Compose([
        A.Lambda(image=lambda x, **kwargs: augraphy_pipeline.augment(x)["output"]),
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])



def get_hybrid_train(img_size=224, augraphy_strength='light'):
    """Augraphy + Albumentations 혼합"""

    # 강도별 확률
    if augraphy_strength == 'light':
        ink_p, paper_p, post_p = 0.2, 0.2, 0.2
    elif augraphy_strength == 'medium':
        ink_p, paper_p, post_p = 0.4, 0.4, 0.3
    else:  # heavy
        ink_p, paper_p, post_p = 0.6, 0.5, 0.4
    
    # Augraphy 파이프라인
    ink_phase = [InkBleed(intensity_range=(0.05, 0.15), p=ink_p)]
    paper_phase = [PaperFactory(p=paper_p), DirtyDrum(p=paper_p * 0.5)]
    post_phase = [
        Jpeg(quality_range=(70, 95), p=post_p),
        Brightness(brightness_range=(0.95, 1.05), p=post_p)
    ]
    
    augraphy_pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)
    
    return A.Compose([
        # Augraphy 적용
        A.Lambda(image=lambda x, **kwargs: augraphy_pipeline.augment(x)["output"]),
        
        # 일반 augmentation
        A.Rotate(limit=3, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(var_limit=(5, 30), p=0.2),
        
        # 전처리
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_augmentation(img_size=224):
    """검증용 augmentation (변환 없음)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_train_augmentation(config):
    """
    Config 기반으로 augmentation 자동 선택
    
    Args:
        config: Config 객체
    """
    strategy = config.AUG_STRATEGY
    dataset_type = config.DATASET_TYPE
    img_size = config.IMAGE_SIZE
    
    # Auto 모드: 데이터셋에 맞게 자동 선택
    if strategy == 'auto':
        if dataset_type in ['cifar10', 'cifar100', 'imagenet']:
            strategy = 'albumentations'
        elif dataset_type in ['document', 'text', 'ocr']:
            strategy = 'hybrid'
        else:
            strategy = 'albumentations'
        
        print(f"📌 Auto mode: {dataset_type} → {strategy} augmentation")
    
    # 전략별 augmentation
    if strategy == 'albumentations':
        return get_albumentations_train(img_size)
    elif strategy == 'augraphy':
        return get_augraphy_train(img_size)
    elif strategy == 'hybrid':
        augraphy_strength = getattr(config, 'AUGRAPHY_STRENGTH', 'light')
        return get_hybrid_train(img_size, augraphy_strength=augraphy_strength)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================
# CIFAR10 Dataset
# ============================================

class CIFAR10Dataset(Dataset):
    """
    CIFAR10 데이터셋을 pickle 파일에서 직접 로드
    
    Args:
        data_dir: CIFAR10 데이터가 있는 디렉토리
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
                    labels = batch.get(b'labels', batch.get('labels', []))
                    batch_data = batch.get(b'data', batch.get('data', None))
                    if batch_data is not None:
                        self.data.append(batch_data)
                        self.labels.extend(labels)
            
            self.data = np.vstack(self.data)  # (50000, 3072)
        else:
            # Test 데이터
            batch_path = self.data_dir / 'test_batch'
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                labels = batch.get(b'labels', batch.get('labels', []))
                batch_data = batch.get(b'data', batch.get('data', None))
                if batch_data is not None:
                    self.data = batch_data
                    self.labels = labels
        
        # 데이터 형태 변환 (3072 -> 32x32x3)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # 특정 인덱스만 사용 (서브샘플링)
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
        image = self.data[idx]  # (32, 32, 3)
        label = int(self.labels[idx])
        
        # Transform 적용
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


# ============================================
# CIFAR10 데이터 로딩 함수
# ============================================

def load_cifar10(config):
    """
    Config 기반으로 CIFAR10 데이터 로드
    
    Args:
        config: Config 객체
        
    Returns:
        train_dataset_raw, test_dataset, train_labels, class_names, num_classes
    """
    print("="*60)
    print("📦 Loading CIFAR10 Data")
    print("="*60)
    
    # 데이터 경로 설정
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_dir = project_root / 'data' / 'cifar-10-batches-py'
    
    # 전체 데이터 로드
    train_data_full = CIFAR10Dataset(
        data_dir=str(data_dir),
        train=True,
        transform=None
    )
    
    test_data_full = CIFAR10Dataset(
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
            class_names = [f'class_{i}' for i in range(10)]
    
    num_classes = len(class_names)
    
    print(f"\n✅ CIFAR10 Full Data Loaded!")
    print(f"Train: {len(train_data_full):,} images")
    print(f"Test:  {len(test_data_full):,} images")
    print(f"Classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # 서브샘플링 (config 기반)
    if config.USE_SUBSET:
        print(f"\n🔥 Subset mode: Using {int(config.SUBSET_RATIO*100)}% of data")
        
        # Train 서브샘플링
        train_labels_full = train_data_full.labels.tolist()
        train_indices = _stratified_subsample(
            train_labels_full, 
            ratio=config.SUBSET_RATIO
        )
        
        train_dataset_raw = CIFAR10Dataset(
            data_dir=str(data_dir),
            train=True,
            transform=None,
            indices=train_indices
        )
        train_labels = [train_labels_full[i] for i in train_indices]
        
        # Test 서브샘플링
        test_labels_full = test_data_full.labels.tolist()
        test_indices = _stratified_subsample(
            test_labels_full,
            ratio=config.SUBSET_RATIO
        )
        
        test_dataset = CIFAR10Dataset(
            data_dir=str(data_dir),
            train=False,
            transform=get_val_augmentation(config.IMAGE_SIZE),
            indices=test_indices
        )
        
        print(f"✅ Subset train size: {len(train_dataset_raw):,}")
        print(f"✅ Subset test size: {len(test_dataset):,}")
    else:
        train_dataset_raw = train_data_full
        train_labels = train_data_full.labels.tolist()
        test_dataset = CIFAR10Dataset(
            data_dir=str(data_dir),
            train=False,
            transform=get_val_augmentation(config.IMAGE_SIZE)
        )
    
    return train_dataset_raw, test_dataset, train_labels, class_names, num_classes


def _stratified_subsample(labels, ratio):
    """클래스별 균등 서브샘플링"""
    indices_by_class = {}
    for idx, label in enumerate(labels):
        if label not in indices_by_class:
            indices_by_class[label] = []
        indices_by_class[label].append(idx)
    
    selected_indices = []
    for label, indices in indices_by_class.items():
        n_samples = int(len(indices) * ratio)
        selected_indices.extend(random.sample(indices, n_samples))
    
    selected_indices.sort()
    return selected_indices


# ============================================
# DataLoader 생성
# ============================================

def get_dataloaders(train_dataset_raw, train_labels, test_dataset, config):
    """
    Config 기반으로 DataLoader 생성
    
    Args:
        train_dataset_raw: Transform 없는 train dataset
        train_labels: Train labels (K-Fold용)
        test_dataset: Transform 적용된 test dataset
        config: Config 객체
        
    Returns:
        train_loader, val_loader (for single split)
        또는 K-Fold에서 직접 사용
    """
    # 이 함수는 K-Fold에서 직접 사용되므로
    # 여기서는 test_loader만 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0  # MPS 호환성
    )
    
    return test_loader


# ============================================
# 전역 변수로 데이터 로드 (backward compatibility)
# ============================================

# Config import
# try:
#     from .config import config
    
#     # 데이터 로드
#     train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_cifar10(config)
    
#     # Device 설정 (config에서)
#     device = config.DEVICE
    
#     # Augmentation
#     train_transform = get_train_augmentation(config)
#     val_transform = get_val_augmentation(config.IMAGE_SIZE)
    
#     print(f"\n🖥️  Device: {device}")
#     print("="*60)
    
# except ImportError:
#     print("⚠️  Config not found. Please import manually.")

# src/data.py 맨 끝에 추가 (기존 코드 뒤에)

# ============================================
# 통합 데이터 로딩 함수
# ============================================

def load_data(config):
    """
    Config 기반으로 데이터 자동 로드
    
    Args:
        config: Config 객체
        
    Returns:
        train_dataset_raw, test_dataset, train_labels, class_names, num_classes
    """
    print(f"\n🎯 Dataset Type: {config.DATASET_TYPE}")
    
    if config.DATASET_TYPE == 'cifar10':
        return load_cifar10(config)
    elif config.DATASET_TYPE == 'cifar100':
        return load_cifar100(config)  # 나중에 필요하면 구현
    elif config.DATASET_TYPE == 'document':
        return load_document_data(config)
    else:
        raise ValueError(
            f"Unknown dataset type: {config.DATASET_TYPE}\n"
            f"Available types: 'cifar10', 'document'"
        )


def load_document_data(config):
    """
    문서 분류 대회 데이터 로드
    
    Args:
        config: Config 객체
        
    Returns:
        train_dataset_raw, test_dataset, train_labels, class_names, num_classes
    """
    print("="*60)
    print("📄 Loading Document Classification Data")
    print("="*60)
    
    # 데이터 경로
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_dir = project_root / 'data' / 'document_competition'
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"❌ Document data not found at {data_dir}\n"
            f"Please download competition data first."
        )
    
    # TODO: 대회 시작하면 아래 구현
    # 
    # 예시 구조:
    # 
    # class DocumentDataset(Dataset):
    #     def __init__(self, data_dir, transform=None):
    #         # CSV 또는 이미지 폴더에서 로드
    #         self.image_paths = list(data_dir.glob('*.jpg'))
    #         self.labels = pd.read_csv(data_dir / 'labels.csv')
    #         self.transform = transform
    #     
    #     def __getitem__(self, idx):
    #         img = cv2.imread(str(self.image_paths[idx]))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         label = self.labels.iloc[idx]['label']
    #         
    #         if self.transform:
    #             augmented = self.transform(image=img)
    #             img = augmented['image']
    #         
    #         return img, label
    # 
    # train_dataset_raw = DocumentDataset(
    #     data_dir / 'train',
    #     transform=None
    # )
    # 
    # test_dataset = DocumentDataset(
    #     data_dir / 'test',
    #     transform=get_val_augmentation(config.IMAGE_SIZE)
    # )
    # 
    # train_labels = train_dataset_raw.labels.tolist()
    # class_names = ['class_0', 'class_1', ...]  # 대회 공지 참고
    # num_classes = len(class_names)
    # 
    # # 서브샘플링 (필요시)
    # if config.USE_SUBSET:
    #     train_indices = _stratified_subsample(train_labels, config.SUBSET_RATIO)
    #     train_dataset_raw = Subset(train_dataset_raw, train_indices)
    #     train_labels = [train_labels[i] for i in train_indices]
    # 
    # return train_dataset_raw, test_dataset, train_labels, class_names, num_classes
    
    raise NotImplementedError(
        "📝 Document dataset loader not implemented yet.\n"
        "Implement this function when competition data is available.\n"
    )
def load_cifar100(config):
    """CIFAR-100 로더 (필요시 구현)"""
    raise NotImplementedError("CIFAR-100 loader not implemented yet.")
# ============================================
# 전역 변수로 데이터 로드 (backward compatibility)
# ============================================

# try:
#     from .config import config
    
#     # load_data() 사용으로 변경! ⭐
#     train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_data(config)
    
#     device = config.DEVICE
#     train_transform = get_train_augmentation(config)
#     val_transform = get_val_augmentation(config.IMAGE_SIZE)
    
#     print(f"\n🖥️  Device: {device}")
#     print("="*60)
    
# except ImportError:
#     print("⚠️  Config not found. Please import manually.")
# except NotImplementedError as e:
#     print(f"⚠️  {e}")