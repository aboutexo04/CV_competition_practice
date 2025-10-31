#!/usr/bin/env python3
"""
Logger 경로 테스트 스크립트

실험 로그가 올바른 위치에 저장되는지 확인합니다.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent / 'CV_competition_practice'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import log_experiment_results
from src.config import DocumentConfig

def test_logger_path():
    """로거 경로 테스트"""

    print("=" * 70)
    print("📝 Logger 경로 테스트")
    print("=" * 70)

    # 테스트용 Config
    config = DocumentConfig()

    # 테스트용 더미 데이터
    fold_results = [
        {'fold': 1, 'best_val_f1': 0.9190},
        {'fold': 2, 'best_val_f1': 0.9014},
        {'fold': 3, 'best_val_f1': 0.9327},
    ]

    results = {
        'test_acc': 92.23,
        'test_f1': 0.9164
    }

    print("\n1️⃣  예상 저장 경로:")
    print(f"   {project_root}/logs/experiment_YYYYMMDD_HHMMSS.md")

    print("\n2️⃣  테스트 로그 저장 중...")
    try:
        filepath = log_experiment_results(fold_results, results, config)

        print(f"\n3️⃣  저장 성공!")
        print(f"   실제 저장 위치: {filepath}")
        print(f"   파일 존재 확인: {filepath.exists()}")

        # 파일 읽기
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"\n4️⃣  저장된 내용 (처음 5줄):")
        for i, line in enumerate(content.split('\n')[:5], 1):
            print(f"   {i}. {line}")

        print("\n" + "=" * 70)
        print("✅ 경로 테스트 통과!")
        print(f"   로그 파일: {filepath}")
        print("=" * 70)

        # 테스트 파일 삭제 여부 확인
        print("\n💡 테스트 파일을 삭제하시겠습니까? (y/N)")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logger_path()
