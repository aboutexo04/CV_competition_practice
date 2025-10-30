"""
Wandb 유틸리티 함수 모듈
"""
import wandb
import os
from datetime import datetime

# Config import (필요시)
try:
    from src.config import MODEL_NAME
    from src.model import MODEL_DISPLAY_NAME, SELECTED_MODEL
except ImportError:
    MODEL_NAME = None
    MODEL_DISPLAY_NAME = None
    SELECTED_MODEL = None

# 🔥 자동 넘버링 함수
def get_next_run_number(project_name, model_name):
    """
    기존 실험들의 넘버링을 확인하여 다음 번호를 반환
    """
    try:
        api = wandb.Api()
        runs = api.runs(f"{api.default_entity}/{project_name}")
        
        # 모델명으로 시작하는 run들의 번호 추출
        existing_numbers = []
        for run in runs:
            if run.name.startswith(model_name):
                # 이름에서 숫자 부분 추출 (예: efficientnet-b0_003 -> 3)
                parts = run.name.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    existing_numbers.append(int(parts[-1]))
        
        # 다음 번호 반환
        next_number = max(existing_numbers) + 1 if existing_numbers else 1
        return next_number
    except:
        # API 접근 실패 시 타임스탬프 기반 번호 반환
        return int(datetime.now().strftime("%H%M%S")) % 1000

def init_wandb_with_auto_numbering(project_name="cifar10-classification-practice",
                                   model_name=None,
                                   model_display_name=None,
                                   config_dict=None):
    """
    자동 넘버링과 함께 Wandb 초기화

    Args:
        project_name: Wandb 프로젝트 이름
        model_name: 모델 이름 (짧은 버전)
        model_display_name: 모델 표시 이름
        config_dict: 실험 설정 딕셔너리

    Returns:
        run: Wandb run 객체
    """
    if model_name is None:
        model_name = MODEL_NAME or "model"
    if model_display_name is None:
        model_display_name = MODEL_DISPLAY_NAME or model_name

    if config_dict is None:
        config_dict = {
            "model": model_display_name,
            "model_architecture": SELECTED_MODEL,
            "dataset": "CIFAR-10",
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.001,
            "n_splits": 2,
            "early_stopping_patience": 5,
            "optimizer": "Adam",
            "image_size": 128,
            "data_subset_ratio": 0.1,
        }

    # 🔥 자동 넘버링된 실험명 생성
    run_number = get_next_run_number(project_name, model_name)
    experiment_name = f"{model_name}_{run_number:03d}"

    print(f"🚀 실험명: {experiment_name}")

    # 실험 시작
    run = wandb.init(
        project=project_name,
        config=config_dict,
        name=experiment_name,
        tags=[model_name, "k-fold", "transfer-learning"]
    )

    print(f"✅ Wandb 초기화 완료! Run ID: {run.id}")
    print(f"📊 대시보드 링크: {run.url}")

    return run
