#!/usr/bin/env python3
"""
Class Weights 테스트 스크립트

Statement of opinion 등 불균형 클래스에 대한 가중치 확인
"""

import sys
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).resolve().parent / 'CV_competition_practice'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_class_weights():
    """Class weights 계산 테스트"""

    print("=" * 70)
    print("⚖️  Class Weights 테스트")
    print("=" * 70)

    # 실제 데이터 분포 (train.csv 기준)
    class_counts = {
        0: 100, 1: 46, 2: 100, 3: 100, 4: 100, 5: 100,
        6: 100, 7: 100, 8: 100, 9: 100, 10: 100, 11: 100,
        12: 100, 13: 74, 14: 50, 15: 100, 16: 100  # 14번이 statement of opinion
    }

    num_classes = 17
    total_samples = sum(class_counts.values())

    print(f"\n📊 데이터 분포:")
    print(f"   Total samples: {total_samples}")
    print(f"   Num classes: {num_classes}")

    # Inverse frequency weighting 계산
    weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    print(f"\n⚖️  계산된 가중치:")
    print(f"   Min weight: {min(weights):.3f}")
    print(f"   Max weight: {max(weights):.3f}")
    print(f"   Average weight: {sum(weights)/len(weights):.3f}")

    print(f"\n🔍 주요 클래스별 가중치:")
    important_classes = [
        (1, "application_for_payment (46개)"),
        (13, "resume (74개)"),
        (14, "statement of opinion (50개)"),  # 가장 적음
        (0, "account_number (100개)")
    ]

    for class_id, name in important_classes:
        weight = weights[class_id]
        count = class_counts[class_id]
        marker = "⚠️ " if count < 60 else "✅ "
        print(f"{marker}Class {class_id:2d} ({name:35s}): weight={weight:.3f}")

    print("\n" + "=" * 70)
    print("💡 효과:")
    print("   - Statement of opinion (50개) → 가중치 높음")
    print("   - 모델이 적은 클래스의 오류에 더 큰 패널티")
    print("   - 불균형 클래스 성능 향상 기대")
    print("=" * 70)

if __name__ == "__main__":
    test_class_weights()
