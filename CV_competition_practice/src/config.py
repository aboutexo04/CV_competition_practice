# src/config.py

import torch

class Config:
    def __init__(self):
        # 데이터셋 설정
        self.DATASET_TYPE = 'document'
        self.IMAGE_SIZE = 224
        self.BATCH_SIZE = 32
        self.USE_SUBSET = False
        self.SUBSET_RATIO = 0.1
        
        # Augmentation 설정
        self.AUG_STRATEGY = 'auto' # albumentations | augraphy | hybrid | auto
        self.AUGRAPHY_STRENGTH = 'light' # light | medium | heavy
        
        # 모델 설정
        self.MODEL_NAME = 'efficientnet_b0'
        self.NUM_CLASSES = 10
        
        # 학습 설정
        self.EPOCHS = 100
        self.LR = 0.001
        self.PATIENCE = 5
        
        # K-Fold 설정
        self.N_FOLDS = 5
        
        # 기타
        self.DEVICE = torch.device(
            'mps' if torch.backends.mps.is_available() 
            else 'cuda' if torch.cuda.is_available() 
            else 'cpu'
        )

        # Wandb
        self.USE_WANDB = False
        self.WANDB_PROJECT = 'document-classification'

        self.USE_TTA = False
        self.TTA_TRANSFORMS = ['original', 'hflip']

        self.USE_LABEL_SMOOTHING = False
        self.LABEL_SMOOTHING_FACTOR = 0.1

        self.USE_ENSEMBLE = True
        self.SEED = 42
        self.DETERMINISTIC = True
        self.NUM_WORKERS = 0

        # Model saving
        self.SAVE_MODEL = True
        self.MODELS_DIR = 'models'
    
    def to_dict(self):
        """Config를 딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def print_config(self):
        """주요 하이퍼파라미터를 깔끔하게 출력"""
        print("=" * 70)
        print("⚙️  실험 설정 (Hyperparameters)")
        print("=" * 70)
        
        print(f"\n📦 데이터셋 설정:")
        print(f"  - 데이터셋:          {self.DATASET_TYPE}")
        print(f"  - 이미지 크기:       {self.IMAGE_SIZE}x{self.IMAGE_SIZE}")
        print(f"  - 서브셋 사용:       {'Yes' if self.USE_SUBSET else 'No'}")
        if self.USE_SUBSET:
            print(f"  - 서브셋 비율:       {self.SUBSET_RATIO * 100:.1f}%")
        
        print(f"\n🎨 Augmentation 설정:")
        print(f"  - 전략:              {self.AUG_STRATEGY}")
        if self.AUG_STRATEGY in ['augraphy', 'hybrid']:
            print(f"  - Augraphy 강도:     {self.AUGRAPHY_STRENGTH}")
        
        print(f"\n🤖 모델 설정:")
        print(f"  - 모델:              {self.MODEL_NAME}")
        print(f"  - 클래스 수:         {self.NUM_CLASSES}")
        
        print(f"\n🎯 학습 설정:")
        print(f"  - Batch Size:        {self.BATCH_SIZE}")
        print(f"  - Epochs:            {self.EPOCHS}")
        print(f"  - Learning Rate:     {self.LR}")
        print(f"  - K-Fold:            {self.N_FOLDS}")
        print(f"  - Early Stop:        {self.PATIENCE} epochs")
        
        print(f"\n🖥️  디바이스:")
        print(f"  - Device:            {self.DEVICE}")
        
        if self.USE_WANDB:
            print(f"\n📊 Wandb:")
            print(f"  - Project:           {self.WANDB_PROJECT}")
        
        print("=" * 70)
    
    def update(self, **kwargs):
        """하이퍼파라미터 동적 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"⚠️  경고: {key}는 Config에 없는 속성입니다.")
        
        print("✅ Config 업데이트 완료!")
        self.print_config()


class QuickTestConfig(Config):
    """빠른 테스트용 설정 (Document 데이터로 빠른 검증)"""
    def __init__(self):
        super().__init__()
        self.DATASET_TYPE = 'document'
        self.EPOCHS = 2
        self.N_FOLDS = 2
        self.BATCH_SIZE = 64
        self.USE_SUBSET = True
        self.SUBSET_RATIO = 0.1
        self.AUG_STRATEGY = 'albumentations'  # 빠른 테스트용
        self.USE_WANDB = False
        self.USE_TTA = False
        self.SAVE_MODEL = False  # Quick test는 모델 저장 안함


class DocumentConfig(Config):
    """문서 분류 대회용 (본격 학습)"""
    def __init__(self):
        super().__init__()
        self.DATASET_TYPE = 'document'
        self.IMAGE_SIZE = 224
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.N_FOLDS = 5
        self.USE_SUBSET = False
        self.AUG_STRATEGY = 'hybrid'
        self.AUGRAPHY_STRENGTH = 'medium'
        self.MODEL_NAME = 'efficientnet_b0'
        self.NUM_CLASSES = 10
        self.LR = 0.0001
        self.PATIENCE = 10
        self.USE_WANDB = False
        self.WANDB_PROJECT = 'document-classification'
        self.USE_TTA = False
        self.SAVE_MODEL = True  # 본격 학습은 best model 저장

# ==========================================
# 기본 config 인스턴스
# ==========================================
config = DocumentConfig()

# 전역 변수로 내보내기 (backward compatibility)
MODEL_NAME = config.MODEL_NAME
BATCH_SIZE = config.BATCH_SIZE
N_FOLDS = config.N_FOLDS
EPOCHS = config.EPOCHS
LR = config.LR
PATIENCE = config.PATIENCE
IMAGE_SIZE = config.IMAGE_SIZE
USE_SUBSET = config.USE_SUBSET
SUBSET_RATIO = config.SUBSET_RATIO
DEVICE = config.DEVICE
device = config.DEVICE
USE_TTA = config.USE_TTA
TTA_TRANSFORMS = config.TTA_TRANSFORMS
# 편의 함수들
def print_config():
    config.print_config()

def update_config(**kwargs):
    config.update(**kwargs)


# ==========================================
# __all__ 정의 (import * 할 때)
# ==========================================
__all__ = [
    # 클래스들
    'Config',
    'QuickTestConfig',
    'DocumentConfig',
    
    # 인스턴스
    'config',
    
    # 전역 변수들
    'MODEL_NAME',
    'BATCH_SIZE',
    'N_FOLDS',
    'EPOCHS',
    'LR',
    'PATIENCE',
    'IMAGE_SIZE',
    'USE_SUBSET',
    'SUBSET_RATIO',
    'DEVICE',
    'device',
    'USE_TTA',
    'TTA_TRANSFORMS',
    # 함수들
    'print_config',
    'update_config',
]