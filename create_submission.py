# create_submission.py (수정 버전)

"""
학습 완료 후 제출 파일 생성 스크립트
"""

import torch
import pandas as pd
from pathlib import Path
import numpy as np
import pickle

from src.config import config
from src.data import load_data
from src.evaluation import evaluate_ensemble


def load_fold_results_from_file(filepath='results/fold_results.pkl'):
    """저장된 fold_results 불러오기"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ fold_results file not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        fold_results = pickle.load(f)
    
    print(f"✅ fold_results loaded from: {filepath}")
    return fold_results


def create_submission_from_fold_results(fold_results, test_dataset, config, use_tta=True):
    """
    Fold 결과로부터 제출 파일 생성
    """
    print("="*70)
    print(f"🎯 Creating Submission File (TTA: {use_tta})")
    print("="*70)
    
    # Evaluation
    test_acc, test_f1, predictions, probs = evaluate_ensemble(
        fold_results=fold_results,
        test_dataset=test_dataset,
        config=config,
        use_tta=use_tta,
        tta_transforms=['original', 'hflip', 'vflip', 'rotate90'] if use_tta else ['original']
    )
    
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    print(f"✅ Test F1 Score: {test_f1:.4f}")
    
    # 제출 파일 생성
    test_image_files = sorted([f.name for f in Path(config.TEST_DIR).glob('*.jpg')])
    
    submission = pd.DataFrame({
        'ID': test_image_files,
        'target': predictions
    })
    
    # 저장
    submission_dir = Path('submissions')
    submission_dir.mkdir(exist_ok=True)
    
    # 파일명에 성능 포함
    if use_tta:
        filename = f'submission_acc{test_acc:.1f}_f1{test_f1:.4f}_TTA.csv'
    else:
        filename = f'submission_acc{test_acc:.1f}_f1{test_f1:.4f}.csv'
    
    submission_path = submission_dir / filename
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission file saved: {submission_path}")
    print(f"   Total predictions: {len(submission)}")
    print("="*70)
    
    return submission_path


def main():
    """메인 실행 함수"""
    
    print("\n" + "="*70)
    print("📝 Submission File Creation")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n1️⃣ Loading data...")
    train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_data(config)
    config.NUM_CLASSES = num_classes
    
    if test_dataset is None:
        print("❌ No test dataset found!")
        return
    
    print(f"✅ Test samples: {len(test_dataset):,}")
    
    # 2. fold_results 불러오기
    print("\n2️⃣ Loading fold results...")
    
    # 방법 A: 메모리에서 (노트북)
    try:
        import __main__
        if hasattr(__main__, 'fold_results'):
            fold_results = __main__.fold_results
            print("✅ fold_results found in memory")
        else:
            raise AttributeError
    except:
        # 방법 B: 파일에서 (터미널)
        print("⚠️  fold_results not in memory, trying to load from file...")
        fold_results = load_fold_results_from_file('results/fold_results.pkl')
        
        if fold_results is None:
            print("\n❌ Could not load fold_results!")
            print("\n💡 해결 방법:")
            print("   1. main_full.py 마지막에 fold_results 저장 코드 추가")
            print("   2. 또는 노트북에서 실행")
            return
    
    # 3. 제출 파일 생성 (TTA 없이)
    print("\n3️⃣ Creating submission (without TTA)...")
    submission_path1 = create_submission_from_fold_results(
        fold_results=fold_results,
        test_dataset=test_dataset,
        config=config,
        use_tta=False
    )
    
    # 4. 제출 파일 생성 (TTA 있음)
    print("\n4️⃣ Creating submission (with TTA)...")
    submission_path2 = create_submission_from_fold_results(
        fold_results=fold_results,
        test_dataset=test_dataset,
        config=config,
        use_tta=True
    )
    
    print("\n" + "="*70)
    print("✅ All Done!")
    print("="*70)
    print(f"Created files:")
    print(f"  1. {submission_path1}")
    print(f"  2. {submission_path2}")
    print("\n💡 다음 단계:")
    print("   - 두 파일 중 F1 score가 높은 것 선택")
    print("   - 대회 사이트에 제출")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()