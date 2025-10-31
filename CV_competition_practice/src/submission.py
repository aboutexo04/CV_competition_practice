# src/submission.py

import pandas as pd
from pathlib import Path
from datetime import datetime


def save_submission(preds, sample_path, save_path, f1_score):
    """
    sample_submission.csv 기반으로 제출 파일 생성

    Args:
        preds: 예측 결과 (numpy array or list)
        sample_path: sample_submission.csv 경로
        save_path: 저장할 파일 경로 (디렉토리 또는 전체 경로)
        f1_score: F1 score 값 (필수, 파일명에 포함)
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
    
    # 저장 경로 생성 (날짜, 시간, F1 score 포함)
    save_path = Path(save_path)

    # 파일명 생성: submission_YYYYMMDD_HHMMSS_F1score.csv
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}_F1{f1_score:.4f}.csv"

    # save_path가 디렉토리면 파일명 추가, 파일이면 디렉토리 사용
    if save_path.suffix == '':  # 디렉토리인 경우
        final_path = save_path / filename
    else:  # 파일 경로인 경우, 부모 디렉토리 사용
        final_path = save_path.parent / filename

    final_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    submission_df.to_csv(final_path, index=False)
    
    print(f"\n✅ Submission 파일 생성 완료!")
    print(f"📁 저장 위치: {final_path}")
    print(f"📊 예측 샘플 수: {len(submission_df):,}")
    print(f"📋 컬럼: {list(submission_df.columns)}")
    print(f"🎯 F1 Score: {f1_score:.4f}")
    print(f"\n미리보기:")
    print(submission_df.head())
    print("=" * 70)

    return submission_df