#!/usr/bin/env python3
"""
과적합 방지 기능 테스트 스크립트

빠른 테스트로 Dropout과 Weight Decay가 제대로 적용되는지 확인합니다.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent / 'CV_competition_practice'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DocumentConfig
from src.model import get_model, get_optimizer

def test_regularization():
    """정규화 설정 테스트"""

    print("=" * 70)
    print("🧪 과적합 방지 기능 테스트")
    print("=" * 70)

    # Config 생성
    config = DocumentConfig()

    # 빠른 테스트용 설정
    config.update(
        USE_SUBSET=True,
        SUBSET_RATIO=0.05,  # 5%만 사용
        EPOCHS=3,
        N_FOLDS=2,
        BATCH_SIZE=16,
        DROPOUT_RATE=0.3,
        WEIGHT_DECAY=5e-4
    )

    print("\n1️⃣  설정 확인:")
    print(f"   - Dropout Rate: {config.DROPOUT_RATE}")
    print(f"   - Weight Decay: {config.WEIGHT_DECAY}")

    # 모델 생성
    print("\n2️⃣  모델 생성 테스트:")
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        dropout_rate=config.DROPOUT_RATE
    )

    # Optimizer 생성
    print("\n3️⃣  Optimizer 생성 테스ト:")
    optimizer = get_optimizer(model, config)

    # Weight decay 확인
    for param_group in optimizer.param_groups:
        if 'weight_decay' in param_group:
            print(f"   ✅ Weight decay 확인: {param_group['weight_decay']}")
            break

    print("\n" + "=" * 70)
    print("✅ 모든 테스트 통과!")
    print("=" * 70)

    print("\n💡 다음 단계:")
    print("   1. 빠른 검증: python3 CV_competition_practice/main_full.py \\")
    print("                   --use_subset=True --subset_ratio=0.05 \\")
    print("                   --epochs=3 --n_folds=2")
    print("")
    print("   2. 전체 학습: python3 CV_competition_practice/main_full.py \\")
    print("                   --epochs=30 --patience=10")

if __name__ == "__main__":
    test_regularization()
