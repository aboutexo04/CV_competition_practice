import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

# Config import
try:
    from src.config import PATIENCE, EPOCHS, BATCH_SIZE, LR, N_FOLDS
except ImportError:
    # 기본값 설정
    PATIENCE = 5
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 0.001
    N_FOLDS = 5


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0):
        """
        Args:
            patience (int): validation F1이 개선되지 않아도 기다릴 epoch 수
            verbose (bool): 메시지 출력 여부
            delta (float): 개선으로 인정할 최소 변화량
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_f1 = 0
        
    def __call__(self, val_f1, model):
        score = val_f1
        
        if self.best_score is None:
            self.best_score = score
            self.best_f1 = val_f1
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⏸️  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_f1 = val_f1
            self.save_checkpoint(val_f1, model)
            self.counter = 0
    
    def save_checkpoint(self, val_f1, model):
        '''validation F1이 개선되면 모델 저장'''
        if self.verbose:
            print(f'✅ Validation F1 improved ({self.best_f1:.4f} → {val_f1:.4f}). Saving model...')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.best_f1 = val_f1


# 🔥 MPS 호환성을 위한 모델 래퍼
class MPSCompatibleModel(nn.Module):
    """
    MPS에서 발생하는 view() 호환성 문제를 해결하는 래퍼
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x):
        # 입력을 contiguous하게 만듦
        x = x.contiguous()
        output = self.base_model(x)
        # 출력도 contiguous하게 만듦
        return output.contiguous()
    
    def load_state_dict(self, state_dict, strict=True):
        return self.base_model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self):
        return self.base_model.state_dict()
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    한 에폭 동안 모델 학습
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스 (CPU/GPU/MPS)
    
    Returns:
        train_loss: 평균 학습 손실
        train_acc: 학습 정확도 (%)
        train_f1: 학습 F1 Score
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_loss = running_loss / total
    train_acc = 100. * correct / total
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return train_loss, train_acc, train_f1


def validate(model, val_loader, criterion, device):
    """
    검증 데이터셋에서 모델 평가
    
    Args:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 디바이스 (CPU/GPU/MPS)
    
    Returns:
        val_loss: 평균 검증 손실
        val_acc: 검증 정확도 (%)
        val_f1: 검증 F1 Score (macro)
        all_preds: 모든 예측 결과
        all_labels: 모든 실제 레이블
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 통계
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    return val_loss, val_acc, val_f1, all_preds, all_labels


def run_kfold_training(model, train_dataset_raw, train_labels, train_transform, val_transform,
                       num_classes, device, config_dict, use_wandb=True):
    """
    K-Fold Cross Validation 학습 실행

    Args:
        model: 기본 모델 (각 fold마다 새로 생성됨)
        train_dataset_raw: 원본 학습 데이터셋
        train_labels: 학습 데이터 레이블
        train_transform: 학습 데이터 augmentation
        val_transform: 검증 데이터 transform
        num_classes: 클래스 수
        device: 디바이스
        config_dict: 설정 딕셔너리 (EPOCHS, BATCH_SIZE, LR, N_FOLDS, PATIENCE 등)
        use_wandb: Wandb 로깅 여부

    Returns:
        fold_results: 각 fold별 결과 리스트
    """
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    import copy

    # Config 추출
    EPOCHS = config_dict.get('EPOCHS', 10)
    BATCH_SIZE = config_dict.get('BATCH_SIZE', 32)
    LR = config_dict.get('LR', 0.001)
    N_FOLDS = config_dict.get('N_FOLDS', 5)
    PATIENCE = config_dict.get('PATIENCE', 5)
    SELECTED_MODEL = config_dict.get('MODEL_NAME', 'efficientnet_b0')

    # K-Fold 설정
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    print("=" * 70)
    print("🚀 K-Fold Cross Validation 시작")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Folds: {N_FOLDS}")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_labels)), train_labels)):
        print(f"\n{'='*70}")
        print(f"📂 Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*70}")

        # DataLoader 생성
        from src.data import CIFAR10Dataset
        train_subset = CIFAR10Dataset(
            data_dir=train_dataset_raw.data_dir,
            train=True,
            transform=train_transform,
            indices=train_idx.tolist()
        )
        val_subset = CIFAR10Dataset(
            data_dir=train_dataset_raw.data_dir,
            train=True,
            transform=val_transform,
            indices=val_idx.tolist()
        )

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # 모델 재생성
        import timm
        fold_model = timm.create_model(SELECTED_MODEL, pretrained=True, num_classes=num_classes)
        fold_model = fold_model.to(device)

        # Optimizer & Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fold_model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=PATIENCE)

        # 학습 이력 저장
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }

        # 학습
        for epoch in range(EPOCHS):
            print(f"\n📍 Epoch [{epoch+1}/{EPOCHS}]")

            train_loss, train_acc, train_f1 = train_one_epoch(fold_model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, _, _ = validate(fold_model, val_loader, criterion, device)

            scheduler.step(val_f1)

            # 이력 저장
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")

            # Wandb 로깅
            if use_wandb:
                import wandb
                wandb.log({
                    f"fold_{fold+1}/train_loss": train_loss,
                    f"fold_{fold+1}/train_acc": train_acc,
                    f"fold_{fold+1}/train_f1": train_f1,
                    f"fold_{fold+1}/val_loss": val_loss,
                    f"fold_{fold+1}/val_acc": val_acc,
                    f"fold_{fold+1}/val_f1": val_f1,
                    f"fold_{fold+1}/epoch": epoch + 1
                })

            early_stopping(val_f1, fold_model)
            if early_stopping.early_stop:
                print("🛑 Early stopping!")
                break

        fold_results.append({
            'fold': fold + 1,
            'best_val_f1': early_stopping.best_f1,
            'best_model_state': early_stopping.best_model_state,
            'history': history
        })

        print(f"\n✅ Fold {fold + 1} 완료! Best Val F1: {early_stopping.best_f1:.4f}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 K-Fold 결과 요약")
    print("=" * 70)
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Val F1 = {result['best_val_f1']:.4f}")

    avg_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in fold_results])
    print("=" * 70)
    print(f"📈 평균 Validation F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print("=" * 70)

    if use_wandb:
        import wandb
        wandb.log({
            "cv/avg_val_f1": avg_f1,
            "cv/std_val_f1": std_f1
        })

    return fold_results

