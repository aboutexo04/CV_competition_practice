# src/config.py

import torch
from pathlib import Path

class Config:
    def __init__(self):
        # 데이터 경로 자동 감지 (실행 위치에 따라 조정)
        self._setup_data_paths()

        # 나머지 설정 초기화
        self._setup_default_config()

    def _setup_data_paths(self):
        """실행 위치에 따라 데이터 경로를 자동으로 설정"""
        current_dir = Path.cwd()

        # 가능한 데이터 경로들
        possible_paths = [
            current_dir / 'data',                    # 프로젝트 루트에서 실행
            current_dir / '../data',                 # CV_competition_practice 폴더에서 실행
            current_dir / '../../data',              # src 폴더에서 실행
        ]

        # 존재하는 경로 찾기
        data_dir = None
        for path in possible_paths:
            if path.exists() and (path / 'train.csv').exists():
                data_dir = path
                break

        if data_dir is None:
            # 기본값 설정 (에러 메시지를 위해)
            data_dir = current_dir / 'data'

        # 경로 설정
        self.DATA_DIR = str(data_dir)
        self.TRAIN_CSV = str(data_dir / 'train.csv')
        self.TEST_DIR = str(data_dir / 'test')
        self.META_CSV = str(data_dir / 'meta.csv')
        self.SUBMISSION_PATH = str(data_dir / 'sample_submission.csv')

    def _setup_default_config(self):
        """기본 설정 초기화"""
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
        self.NUM_CLASSES = 17  # meta.csv에 0~16까지 17개 클래스

        # 정규화 설정 (Overfitting Prevention)
        self.DROPOUT_RATE = 0.3  # Dropout 비율 (0.0 ~ 0.5 권장)
        self.WEIGHT_DECAY = 5e-4  # L2 regularization (1e-4 ~ 1e-3 권장)

        # 학습 설정
        self.EPOCHS = 100
        self.LR = 0.001
        self.PATIENCE = 5
        self.EARLY_STOPPING_DELTA = 0  # Early stopping delta (0 = accept any improvement)

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

        # ============================================
        # Class Imbalance 대응 설정
        # ============================================
        self.USE_CLASS_WEIGHTS = False
        self.CLASS_WEIGHT_POWER = 0.7  # 0.5 (완화) ~ 1.0 (강함)
        self.MANUAL_CLASS_WEIGHTS = {}  # 예: {8: 3.0} - 특정 클래스 가중치 수동 설정
        
        # Statement of Opinion 클래스 분석
        self.OPINION_CLASS_ID = None  # TODO: 실제 클래스 ID 확인 후 입력 (0~16 중 하나)

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
        print(f"  - Early Stop:        {self.PATIENCE} epochs (delta={self.EARLY_STOPPING_DELTA})")

        print(f"\n🛡️  정규화 설정:")
        print(f"  - Dropout Rate:      {self.DROPOUT_RATE}")
        print(f"  - Weight Decay:      {self.WEIGHT_DECAY}")
        
        print(f"\n⚖️  Class Imbalance 대응:")
        print(f"  - Class Weights:     {'Yes' if self.USE_CLASS_WEIGHTS else 'No'}")
        if self.USE_CLASS_WEIGHTS:
            print(f"  - Weight Power:      {self.CLASS_WEIGHT_POWER}")
            if self.MANUAL_CLASS_WEIGHTS:
                print(f"  - Manual Weights:    {self.MANUAL_CLASS_WEIGHTS}")
        if self.OPINION_CLASS_ID is not None:
            print(f"  - Opinion Class ID:  {self.OPINION_CLASS_ID} (자동 분석 활성화)")
        
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
        self.EPOCHS = 20
        self.N_FOLDS = 5
        self.BATCH_SIZE = 32
        self.USE_SUBSET = False
        self.SUBSET_RATIO = 0.1
        self.AUG_STRATEGY = 'albumentations'  # 빠른 테스트용
        self.USE_WANDB = False
        self.USE_TTA = False
        self.SAVE_MODEL = False  # Quick test는 모델 저장 안함
        self.IMAGE_SIZE = 224
        self.LR = 0.0001
        
        # Class Imbalance - Quick Test에서는 비활성화
        self.USE_CLASS_WEIGHTS = False
        self.OPINION_CLASS_ID = None


class DocumentConfig(Config):
    """문서 분류 대회용 (본격 학습)"""
    def __init__(self):
        super().__init__()
        self.DATASET_TYPE = 'document'
        self.IMAGE_SIZE = 224
        self.BATCH_SIZE = 32
        self.EPOCHS = 20
        self.N_FOLDS = 5
        self.USE_SUBSET = False
        self.AUG_STRATEGY = 'hybrid'
        self.AUGRAPHY_STRENGTH = 'medium'
        self.MODEL_NAME = 'tf_efficientnetv2_s'
        self.NUM_CLASSES = 17  # meta.csv에 0~16까지 17개 클래스
        self.LR = 0.001
        self.PATIENCE = 10
        self.USE_WANDB = False
        self.WANDB_PROJECT = 'document-classification'
        self.USE_TTA = True
        self.SAVE_MODEL = True  # 본격 학습은 best model 저장
        
        # ============================================
        # Class Imbalance 대응 (Statement of Opinion)
        # ============================================
        self.USE_CLASS_WEIGHTS = True
        self.CLASS_WEIGHT_POWER = 0.7  # 처음엔 0.7로 시작 (적당한 강도)
        self.MANUAL_CLASS_WEIGHTS = {}  # 분석 후 필요시 설정: {클래스ID: 배수}
        
        # TODO: meta.csv에서 "statement of opinion" 클래스의 실제 ID 확인 후 입력
        # 예: self.OPINION_CLASS_ID = 8
        self.OPINION_CLASS_ID = None  # 0~16 중 하나


class OpinionFocusedConfig(DocumentConfig):
    """Statement of Opinion 클래스에 집중한 설정"""
    def __init__(self):
        super().__init__()
        
        # Opinion 클래스 강화 설정
        self.USE_CLASS_WEIGHTS = True
        self.CLASS_WEIGHT_POWER = 0.7
        
        # TODO: Opinion 클래스 ID 확인 후 설정
        # 예: self.OPINION_CLASS_ID = 8
        # 예: self.MANUAL_CLASS_WEIGHTS = {8: 3.0}
        self.OPINION_CLASS_ID = None  # 실제 ID로 변경 필요
        self.MANUAL_CLASS_WEIGHTS = {}  # 분석 후 설정: {OPINION_CLASS_ID: 3.0}
        
        # 학습 설정 조정
        self.PATIENCE = 15  # 더 오래 기다림
        self.EARLY_STOPPING_DELTA = 0.001  # 미세한 개선도 인정


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
    'OpinionFocusedConfig',
    
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