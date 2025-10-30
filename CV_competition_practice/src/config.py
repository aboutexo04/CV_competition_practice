import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import copy
from pathlib import Path
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    def __init__(self):
        self.MODEL_NAME = 'efficientnet_b0'
        self.BATCH_SIZE = 32
        self.N_FOLDS = 2
        self.EPOCHS = 1
        self.LR = 0.001
        self.PATIENCE=5
        self.IMAGE_SIZE=224
        self.USE_SUBSET = True
        self.SUBSET_RATIO = 0.1

config = Config()

# 전역 변수로 내보내기
MODEL_NAME = config.MODEL_NAME
BATCH_SIZE = config.BATCH_SIZE
N_FOLDS = config.N_FOLDS
EPOCHS = config.EPOCHS
LR = config.LR
PATIENCE = config.PATIENCE
IMAGE_SIZE = config.IMAGE_SIZE
USE_SUBSET = config.USE_SUBSET
SUBSET_RATIO = config.SUBSET_RATIO

# Device 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


def print_config():
    """주요 하이퍼파라미터를 깔끔하게 출력"""
    print("=" * 70)
    print("⚙️  실험 설정 (Hyperparameters)")
    print("=" * 70)
    print(f"\n📦 모델 설정:")
    print(f"  - 모델:              {MODEL_NAME}")
    print(f"  - 이미지 크기:       {IMAGE_SIZE}x{IMAGE_SIZE}")

    print(f"\n🎯 학습 설정:")
    print(f"  - Batch Size:        {BATCH_SIZE}")
    print(f"  - Epochs:            {EPOCHS}")
    print(f"  - Learning Rate:     {LR}")
    print(f"  - K-Fold:            {N_FOLDS}")
    print(f"  - Early Stop:        {PATIENCE} epochs")

    print(f"\n💾 데이터 설정:")
    print(f"  - 서브셋 사용:       {'Yes' if USE_SUBSET else 'No'}")
    if USE_SUBSET:
        print(f"  - 서브셋 비율:       {SUBSET_RATIO * 100:.1f}%")

    print(f"\n🖥️  디바이스:")
    print(f"  - Device:            {device}")
    print("=" * 70)


def update_config(**kwargs):
    """
    하이퍼파라미터 동적 업데이트

    Examples:
        update_config(BATCH_SIZE=64, EPOCHS=20, LR=0.0001)
    """
    global MODEL_NAME, BATCH_SIZE, N_FOLDS, EPOCHS, LR, PATIENCE, IMAGE_SIZE, USE_SUBSET, SUBSET_RATIO

    if 'MODEL_NAME' in kwargs:
        MODEL_NAME = kwargs['MODEL_NAME']
    if 'BATCH_SIZE' in kwargs:
        BATCH_SIZE = kwargs['BATCH_SIZE']
    if 'N_FOLDS' in kwargs:
        N_FOLDS = kwargs['N_FOLDS']
    if 'EPOCHS' in kwargs:
        EPOCHS = kwargs['EPOCHS']
    if 'LR' in kwargs:
        LR = kwargs['LR']
    if 'PATIENCE' in kwargs:
        PATIENCE = kwargs['PATIENCE']
    if 'IMAGE_SIZE' in kwargs:
        IMAGE_SIZE = kwargs['IMAGE_SIZE']
    if 'USE_SUBSET' in kwargs:
        USE_SUBSET = kwargs['USE_SUBSET']
    if 'SUBSET_RATIO' in kwargs:
        SUBSET_RATIO = kwargs['SUBSET_RATIO']

    print("✅ Config 업데이트 완료!")
    print_config()


print(config.__dict__)
