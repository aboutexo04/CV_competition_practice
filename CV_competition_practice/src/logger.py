# src/logger.py

"""
실험 결과 로깅 및 저장 모듈
"""
from datetime import datetime
from pathlib import Path


def log_experiment_results(fold_results, results, config):
    """
    실험 결과를 마크다운 파일로 간단히 저장
    
    Args:
        fold_results: K-Fold 학습 결과
        results: 테스트 결과
        config: Config 객체
    """
    # 저장 경로
    project_root = Path.cwd().parent
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"experiment_{timestamp}.md"
    filepath = log_dir / filename
    
    # 마크다운 내용 생성
    content = f"""# 실험 결과

## 실험 정보
- **날짜**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **모델**: {config.MODEL_NAME}
- **데이터셋**: {config.DATASET_TYPE}

## 하이퍼파라미터
- **Epochs**: {config.EPOCHS}
- **Batch Size**: {config.BATCH_SIZE}
- **Learning Rate**: {config.LR}
- **N-Folds**: {config.N_FOLDS}
- **Image Size**: {config.IMAGE_SIZE}

## K-Fold 결과
"""
    
    # Fold별 결과
    for r in fold_results:
        content += f"- **Fold {r['fold']}**: Val F1 = {r['best_val_f1']:.4f}\n"
    
    # 평균 계산
    import numpy as np
    avg_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in fold_results])
    
    content += f"\n**평균 Val F1**: {avg_f1:.4f} ± {std_f1:.4f}\n"
    
    # 테스트 결과
    content += f"""
## 테스트 결과
- **Accuracy**: {results['test_acc']:.2f}%
- **F1 Score**: {results['test_f1']:.4f}
"""
    
    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ 실험 결과 저장: {filepath}")
    
    return filepath