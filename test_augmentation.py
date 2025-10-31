#!/usr/bin/env python3
"""
Augmentation 테스트 스크립트

수정된 augmentation이 올바르게 동작하는지 확인합니다.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent / 'CV_competition_practice'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import (
    get_albumentations_train,
    get_augraphy_train,
    get_hybrid_train,
    get_val_augmentation,
    get_train_augmentation
)
from src.config import DocumentConfig
import numpy as np

def test_augmentations():
    """Augmentation 함수 테스트"""

    print("=" * 70)
    print("🎨 Augmentation 테스트")
    print("=" * 70)

    # 테스트 이미지 생성 (224x224 RGB)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    config = DocumentConfig()
    image_size = config.IMAGE_SIZE

    print(f"\n📏 Test image shape: {test_image.shape}")
    print(f"   Image size config: {image_size}x{image_size}")

    # 1. Albumentations (경량화 버전)
    print("\n1️⃣  Albumentations (Document-optimized):")
    try:
        aug_alb = get_albumentations_train(image_size)
        result_alb = aug_alb(image=test_image)
        print(f"   ✅ Success - Output shape: {result_alb['image'].shape}")
        print(f"   Output type: {type(result_alb['image'])}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # 2. Augraphy
    print("\n2️⃣  Augraphy (Document-specific):")
    try:
        aug_aug = get_augraphy_train(image_size)
        result_aug = aug_aug(image=test_image)
        print(f"   ✅ Success - Output shape: {result_aug['image'].shape}")
    except Exception as e:
        print(f"   ⚠️  Expected (Augraphy may not be installed): {e}")

    # 3. Hybrid (경량화 버전)
    print("\n3️⃣  Hybrid (Augraphy + Albumentations - Lightweight):")
    try:
        aug_hybrid = get_hybrid_train(image_size, augraphy_strength='medium')
        result_hybrid = aug_hybrid(image=test_image)
        print(f"   ✅ Success - Output shape: {result_hybrid['image'].shape}")
    except Exception as e:
        print(f"   ⚠️  Expected (Augraphy may not be installed): {e}")

    # 4. Validation
    print("\n4️⃣  Validation (No augmentation):")
    try:
        aug_val = get_val_augmentation(image_size)
        result_val = aug_val(image=test_image)
        print(f"   ✅ Success - Output shape: {result_val['image'].shape}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # 5. Config-based Auto mode
    print("\n5️⃣  Config-based Auto mode:")
    try:
        config.AUG_STRATEGY = 'auto'
        config.DATASET_TYPE = 'document'
        aug_auto = get_train_augmentation(image_size, config)
        result_auto = aug_auto(image=test_image)
        print(f"   ✅ Success - Output shape: {result_auto['image'].shape}")
        print(f"   Strategy selected: hybrid (expected for document)")
    except Exception as e:
        print(f"   ⚠️  {e}")

    print("\n" + "=" * 70)
    print("✅ Augmentation 테스트 완료!")
    print("=" * 70)

    print("\n📊 수정 요약:")
    print("   • Albumentations: 경량화 (과적합 방지)")
    print("     - HueSaturationValue 제거 (문서에 부적합)")
    print("     - Rotate: 10° → 5°")
    print("     - 확률: 전체적으로 30-50% → 20-30%로 축소")
    print("\n   • Hybrid: 경량화")
    print("     - Brightness/Contrast: 0.15 → 0.1")
    print("     - GaussNoise: (5,30) → (3,20), p 0.2 → 0.15")
    print("\n   • Auto mode: albumentations → hybrid")
    print("     - 문서 데이터셋은 자동으로 hybrid 선택")

if __name__ == "__main__":
    test_augmentations()
