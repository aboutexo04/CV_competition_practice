# src/data.py

import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from src.config import config

# ============================================
# Augmentation 함수들
# ============================================

def get_albumentations_train(image_size):
    """일반 이미지용 augmentation - 강화 버전"""

    return A.Compose([
        A.Resize(image_size, image_size),

        # 기하학적 변환 강화
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=10,
            p=0.5
        ),

        # 색상/명암 변환 추가
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.4
        ),

        # 노이즈 및 블러 추가
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # 품질 저하 시뮬레이션
        A.OneOf([
            A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),

        # 정규화
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


def get_augraphy_train(image_size):  # ✅ 파라미터명 통일
    """문서 특화 augmentation (Augraphy)"""
    if not AUGRAPHY_AVAILABLE:
        print("⚠️  Augraphy not installed. Falling back to Albumentations.")
        return get_albumentations_train(image_size)  # ✅ 수정
    
    import numpy as np
    
    # Augraphy 파이프라인
    ink_phase = [
        InkBleed(intensity_range=(0.1, 0.3), p=0.2),
    ]
    
    paper_phase = [
        PaperFactory(p=0.2),
        DirtyDrum(p=0.1),
    ]
    
    post_phase = [
        Jpeg(quality_range=(60, 95), p=0.2),
        Brightness(brightness_range=(0.95, 1.05), p=0.2),
    ]
    
    augraphy_pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)
    
    # ✅ Augraphy 적용 + 채널 보정 함수
    def apply_augraphy_safe(image, **kwargs):
        """Augraphy 적용 후 RGB 채널 보장"""
        # Augraphy 적용
        result = augraphy_pipeline.augment(image)["output"]
        
        # 채널 수 확인 및 보정
        if len(result.shape) == 2:  # Grayscale (H, W)
            result = np.stack([result] * 3, axis=-1)  # (H, W, 3)
        elif result.shape[-1] == 1:  # (H, W, 1)
            result = np.repeat(result, 3, axis=-1)  # (H, W, 3)
        elif result.shape[-1] == 4:  # RGBA (H, W, 4)
            result = result[:, :, :3]  # RGB만 (H, W, 3)
        
        return result
    
    return A.Compose([
        A.Lambda(image=apply_augraphy_safe),
        A.Resize(image_size, image_size),  # ✅ 수정
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_hybrid_train(image_size, augraphy_strength='light'):  # ✅ 파라미터명 통일
    """Augraphy + Albumentations 혼합"""
    
    if not AUGRAPHY_AVAILABLE:
        print("⚠️  Augraphy not installed. Falling back to Albumentations.")
        return get_albumentations_train(image_size)  # ✅ 수정

    import numpy as np

    # 강도별 확률
    if augraphy_strength == 'light':
        ink_p, paper_p, post_p = 0.2, 0.2, 0.2
    elif augraphy_strength == 'medium':
        ink_p, paper_p, post_p = 0.4, 0.4, 0.3
    elif augraphy_strength == 'heavy':  # heavy
        ink_p, paper_p, post_p = 0.6, 0.5, 0.4
    
    # Augraphy 파이프라인
    ink_phase = [InkBleed(intensity_range=(0.05, 0.15), p=ink_p)]
    paper_phase = [PaperFactory(p=paper_p), DirtyDrum(p=paper_p * 0.5)]
    post_phase = [
        Jpeg(quality_range=(70, 95), p=post_p),
        Brightness(brightness_range=(0.95, 1.05), p=post_p)
    ]
    
    augraphy_pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)
    
    # ✅ 채널 보정 함수
    def apply_augraphy_safe(image, **kwargs):
        result = augraphy_pipeline.augment(image)["output"]
        
        if len(result.shape) == 2:
            result = np.stack([result] * 3, axis=-1)
        elif result.shape[-1] == 1:
            result = np.repeat(result, 3, axis=-1)
        elif result.shape[-1] == 4:
            result = result[:, :, :3]
        
        return result
    
    return A.Compose([
        # Augraphy 적용
        A.Lambda(image=apply_augraphy_safe),

        # 일반 augmentation
        A.Rotate(limit=3, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(var_limit=(5, 30), p=0.2),

        # 전처리
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_augmentation(image_size):  # ✅ 파라미터명 통일
    """검증용 augmentation (변환 없음)"""
    return A.Compose([
        A.Resize(image_size, image_size),  # ✅ 수정
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_train_augmentation(image_size, config):  # ✅ 시그니처 수정
    """
    Config 기반으로 augmentation 자동 선택
    
    Args:
        image_size: 이미지 크기
        config: Config 객체
    """
    # Config에서 속성 가져오기 (없으면 기본값)
    strategy = getattr(config, 'AUG_STRATEGY', 'auto')
    dataset_type = getattr(config, 'DATASET_TYPE', 'document')
    
    # Auto 모드: 데이터셋에 맞게 자동 선택
    if strategy == 'auto':
        if dataset_type in ['document', 'text', 'ocr']:
            strategy = 'hybrid'  # 문서는 hybrid (Augraphy + Albumentations)
        else:
            strategy = 'albumentations'  # 일반 이미지

        print(f"📌 Auto mode: {dataset_type} → {strategy} augmentation")
    
    # 전략별 augmentation
    if strategy == 'albumentations':
        return get_albumentations_train(image_size)
    elif strategy == 'augraphy':
        return get_augraphy_train(image_size)
    elif strategy == 'hybrid':
        augraphy_strength = getattr(config, 'AUGRAPHY_STRENGTH', 'light')
        return get_hybrid_train(image_size, augraphy_strength=augraphy_strength)
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

    # Config에서 데이터 경로 가져오기
    train_csv_path = Path(config.TRAIN_CSV)
    meta_csv_path = Path(config.META_CSV)
    train_img_dir = Path(config.DATA_DIR) / 'train'
    test_img_dir = Path(config.TEST_DIR)
    
    # CSV 파일 존재 확인
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"❌ train.csv not found at {train_csv_path}\n"
            f"Please check config.TRAIN_CSV path"
        )

    if not meta_csv_path.exists():
        raise FileNotFoundError(
            f"❌ meta.csv not found at {meta_csv_path}\n"
            f"Please check config.META_CSV path"
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

    # Test 데이터셋 로드 (test 디렉토리가 있는 경우)
    if test_img_dir.exists():
        # test 이미지 파일 목록 가져오기
        test_image_files = sorted([f.name for f in test_img_dir.glob('*.jpg')])

        # test용 DataFrame 생성 (ID만 있고 target은 없음)
        test_df = pd.DataFrame({'ID': test_image_files})
        test_df['target'] = -1  # placeholder

        # Test dataset 생성 (validation transform 사용)
        test_transform = get_val_augmentation(config.IMAGE_SIZE)

        # Test용 DocumentDataset (transform 적용)
        class DocumentTestDataset(Dataset):
            def __init__(self, df, img_dir, transform):
                self.img_dir = Path(img_dir)
                self.transform = transform
                self.image_ids = df['ID'].tolist()

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                img_id = self.image_ids[idx]
                img_path = self.img_dir / img_id

                try:
                    image = Image.open(img_path).convert('RGB')
                    image = np.array(image)
                except Exception as e:
                    print(f"⚠️  Error loading image {img_path}: {e}")
                    image = np.zeros((224, 224, 3), dtype=np.uint8)

                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']

                return image, -1  # label은 -1 (placeholder)

        test_dataset = DocumentTestDataset(
            df=test_df,
            img_dir=test_img_dir,
            transform=test_transform
        )

        print(f"Test samples: {len(test_dataset):,}")
    else:
        test_dataset = None
        print("Test directory not found - test_dataset set to None")

    return train_dataset_raw, test_dataset, train_labels, class_names, num_classes


def load_data(config):
    """
    Config 기반으로 데이터 자동 로드
    
    Args:
        config: Config 객체
        
    Returns:
        train_dataset_raw, test_dataset, train_labels, class_names, num_classes
    """
    print(f"\n🎯 Dataset Type: {getattr(config, 'DATASET_TYPE', 'document')}")
    
    dataset_type = getattr(config, 'DATASET_TYPE', 'document')
    
    if dataset_type == 'document':
        return load_document_data(config)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}\n"
            f"Available types: 'document'"
        )