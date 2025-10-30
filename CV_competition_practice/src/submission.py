# src/submission.py

import pandas as pd
from pathlib import Path


def save_submission(preds, sample_path, save_path):
    """
    sample_submission.csv 기반으로 제출 파일 생성
    
    Args:
        preds: 예측 결과 (numpy array or list)
        sample_path: sample_submission.csv 경로
        save_path: 저장할 파일 경로
    """
    print("=" * 70)
    print("📝 Submission 파일 생성 중...")
    print("=" * 70)
    
    # Sample submission 로드
    sample_path = Path(sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(
            f"❌ sample_submission.csv를 찾을 수 없습니다: {sample_path}\n"
            f"대회 페이지에서 sample_submission.csv를 다운로드 후 프로젝트 루트에 배치해주세요."
        )
    
    sample_df = pd.read_csv(sample_path)
    
    # 예측값 개수 확인
    if len(preds) != len(sample_df):
        raise ValueError(
            f"❌ 예측값 개수 불일치!\n"
            f"예측값: {len(preds)}, sample_submission: {len(sample_df)}"
        )
    
    # 제출 파일 생성
    submission_df = sample_df.copy()
    submission_df['target'] = preds
    
    # 저장 경로 생성
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장
    submission_df.to_csv(save_path, index=False)
    
    print(f"\n✅ Submission 파일 생성 완료!")
    print(f"📁 저장 위치: {save_path}")
    print(f"📊 예측 샘플 수: {len(submission_df):,}")
    print(f"📋 컬럼: {list(submission_df.columns)}")
    print(f"\n미리보기:")
    print(submission_df.head())
    print("=" * 70)
    
    return submission_df