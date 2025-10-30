# src/data.py

import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image

# ============================================
# Augmentation 함수들
# ============================================

def get_albumentations_train(img_size=224):
    """일반 이미지용 augmentation"""
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
    
    # Augraphy 파이프라인
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
    
    if not AUGRAPHY_AVAILABLE:
        print("⚠️  Augraphy not installed. Falling back to Albumentations.")
        return get_albumentations_train(img_size)

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
        if dataset_type in ['document', 'text', 'ocr']:
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
# Document Dataset 클래스
# ============================================

class DocumentDataset(Dataset):
    """
    문서 분류 대회용 Dataset
    
    Args:
        df: train.csv DataFrame (ID, target 컬럼)
        img_dir: 이미지 디렉토리 경로
        transform: Albumentations transform
        indices: 사용할 인덱스 리스트 (K-Fold용, None이면 전체)
    """
    def __init__(self, df, img_dir, transform=None, indices=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        if indices is not None:
            self.df = df.iloc[indices].reset_index(drop=True)
            self.indices = indices
        else:
            self.df = df
            self.indices = list(range(len(df)))
        
        self.image_ids = self.df['ID'].tolist()
        self.labels = self.df['target'].tolist()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_id = self.image_ids[idx]
        img_path = self.img_dir / img_id
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"⚠️  Error loading image {img_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        label = int(self.labels[idx])
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


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
        test_loader (K-Fold에서 직접 train_loader 생성)
    """
    # test_dataset이 None이면 (대회 초기) None 반환
    if test_dataset is None:
        return None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0  # MPS 호환성
    )
    
    return test_loader


# ============================================
# 데이터 로딩 함수
# ============================================

def load_document_data(config):
    """
    문서 분류 대회 데이터 로드
    
    데이터 구조:
    - train.csv: ID, target (이미지명 <-> 클래스 인덱스)
    - meta.csv: target, class_name (클래스 인덱스 <-> 클래스명)
    - train/: 학습 이미지 디렉토리
    
    Args:
        config: Config 객체
        
    Returns:
        train_dataset_raw, test_dataset, train_labels, class_names, num_classes
    """
    print("="*60)
    print("📄 Loading Document Classification Data")
    print("="*60)
    
    # 데이터 경로 설정
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_dir = project_root / 'data' / 'document'
    
    train_csv_path = data_dir / 'train.csv'
    meta_csv_path = data_dir / 'meta.csv'
    train_img_dir = data_dir / 'train'
    
    # CSV 파일 존재 확인
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"❌ train.csv not found at {train_csv_path}\n"
            f"Please download competition data and place it in {data_dir}"
        )
    
    if not meta_csv_path.exists():
        raise FileNotFoundError(
            f"❌ meta.csv not found at {meta_csv_path}\n"
            f"Please download competition data and place it in {data_dir}"
        )
    
    # CSV 로드
    train_df = pd.read_csv(train_csv_path)
    meta_df = pd.read_csv(meta_csv_path)
    
    # 클래스 정보 추출
    meta_df = meta_df.sort_values('target').reset_index(drop=True)
    class_names = meta_df['class_name'].tolist()
    num_classes = len(class_names)
    
    print(f"\n✅ Document Data Loaded!")
    print(f"Total samples: {len(train_df):,}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names[:5]}{'...' if num_classes > 5 else ''}")
    
    # Dataset 생성
    train_dataset_raw = DocumentDataset(
        df=train_df,
        img_dir=train_img_dir,
        transform=None
    )
    train_labels = train_df['target'].tolist()
    
    # Test 데이터는 나중에 제공되면 추가
    test_dataset = None
    
    return train_dataset_raw, test_dataset, train_labels, class_names, num_classes


def load_data(config):
    """
    Config 기반으로 데이터 자동 로드
    
    Args:
        config: Config 객체
        
    Returns:
        train_dataset_raw, test_dataset, train_labels, class_names, num_classes
    """
    print(f"\n🎯 Dataset Type: {config.DATASET_TYPE}")
    
    if config.DATASET_TYPE == 'document':
        return load_document_data(config)
    else:
        raise ValueError(
            f"Unknown dataset type: {config.DATASET_TYPE}\n"
            f"Available types: 'document'"
        )