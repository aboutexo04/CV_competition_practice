# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from src.model import LabelSmoothingLoss


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
        self.best_model_state = None
        
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
            print(f'✅ Validation F1 improved ({self.best_f1:.4f} → {val_f1:.4f})')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.best_f1 = val_f1


class TransformSubset(torch.utils.data.Dataset):
    """Transform을 적용하는 Subset wrapper"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img, label


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    한 에폭 동안 모델 학습
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
    
    Returns:
        train_loss, train_acc, train_f1
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
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
        device: 디바이스
    
    Returns:
        val_loss, val_acc, val_f1, all_preds, all_labels
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
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


def run_kfold_training(train_dataset_raw, train_labels, config):
    """
    K-Fold Cross Validation 학습 실행 (Config 기반)
    
    Args:
        train_dataset_raw: Transform이 적용되지 않은 원본 dataset
        train_labels: Train labels
        config: Config 객체
        
    Returns:
        fold_results: 각 fold별 결과 리스트
    """
    from src.data import get_train_augmentation, get_val_augmentation
    from src.model import get_model, get_optimizer
    
    # Config에서 설정 가져오기
    device = config.DEVICE
    num_classes = config.NUM_CLASSES
    use_wandb = config.USE_WANDB
    n_folds = config.N_FOLDS
    epochs = config.EPOCHS
    batch_size = config.BATCH_SIZE
    model_name = config.MODEL_NAME
    patience = config.PATIENCE
    
    print("=" * 70)
    print("🚀 K-Fold Cross Validation 시작")
    print(f"Model: {model_name}, Epochs: {epochs}, Batch: {batch_size}, Folds: {n_folds}")
    print("=" * 70)
    
    # Augmentation
    train_transform = get_train_augmentation(config)
    val_transform = get_val_augmentation(config.IMAGE_SIZE)
    
    # K-Fold 설정
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset_raw, train_labels)):
        print(f"\n{'='*70}")
        print(f"📁 Fold {fold + 1}/{n_folds}")
        print(f"{'='*70}")
        
        # Dataset & DataLoader
        train_subset = Subset(train_dataset_raw, train_idx)
        val_subset = Subset(train_dataset_raw, val_idx)
        
        # Transform 적용
        train_dataset = TransformSubset(train_subset, train_transform)
        val_dataset = TransformSubset(val_subset, val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # 모델 생성 (매 fold마다 새로 생성)
        model = get_model(model_name, num_classes, pretrained=True).to(device)
        optimizer = get_optimizer(model, config)
        # Loss 함수 선택
        if config.USE_LABEL_SMOOTHING:
            criterion = LabelSmoothingLoss(
                num_classes=num_classes,
                smoothing=config.LABEL_SMOOTHING_FACTOR
            )
            if fold == 0:  # 첫 번째 fold에서만 출력
                print(f"✅ Label Smoothing 사용 (smoothing={config.LABEL_SMOOTHING_FACTOR})")
        else:
            criterion = nn.CrossEntropyLoss()
            if fold == 0:
                print("✅ CrossEntropyLoss 사용")
        
        # Scheduler & Early Stopping
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # 학습 이력
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        # 학습 루프
        for epoch in range(epochs):
            print(f"\n📍 Epoch [{epoch+1}/{epochs}]")
            
            # Train
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validation
            val_loss, val_acc, val_f1, _, _ = validate(
                model, val_loader, criterion, device
            )
            
            # Scheduler step
            scheduler.step(val_f1)
            
            # 이력 저장
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # 출력
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
            
            # Early Stopping
            early_stopping(val_f1, model)
            if early_stopping.early_stop:
                print("🛑 Early stopping!")
                break
        
        # Fold 결과 저장
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
    
    # Wandb 최종 로깅
    if use_wandb:
        import wandb
        wandb.log({
            "cv/avg_val_f1": avg_f1,
            "cv/std_val_f1": std_f1
        })
    
    return fold_results