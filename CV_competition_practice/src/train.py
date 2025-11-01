# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter
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


def safe_collate_fn(batch):
    """채널 수 안전 장치"""
    import torch
    
    images = []
    labels = []
    
    for image, label in batch:
        # 채널 확인
        if image.ndim == 2:  # (H, W)
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[0] == 1:  # (1, H, W)
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:  # (4, H, W) - RGBA
            image = image[:3, :, :]
        
        images.append(image)
        labels.append(label)
    
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels


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


def analyze_opinion_class(val_labels, val_preds, opinion_class_id, num_classes):
    """
    Statement of Opinion 클래스 상세 분석
    
    Args:
        val_labels: 실제 레이블
        val_preds: 예측 레이블
        opinion_class_id: Statement of Opinion 클래스 ID
        num_classes: 전체 클래스 수
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from collections import Counter
    
    print("\n" + "="*70)
    print("💭 Statement of Opinion 상세 분석")
    print("="*70)
    
    # 기본 통계
    val_labels_np = np.array(val_labels)
    val_preds_np = np.array(val_preds)
    
    opinion_actual_mask = val_labels_np == opinion_class_id
    opinion_preds_mask = val_preds_np == opinion_class_id
    
    print(f"실제 Opinion 샘플: {opinion_actual_mask.sum()}개")
    print(f"예측된 Opinion: {opinion_preds_mask.sum()}개")
    
    # Precision, Recall, F1
    p, r, f1, support = precision_recall_fscore_support(
        val_labels, val_preds, 
        labels=[opinion_class_id],
        zero_division=0
    )
    
    print(f"\n📊 성능 지표:")
    print(f"   Precision: {p[0]:.3f} (예측한 Opinion 중 맞춘 비율)")
    print(f"   Recall:    {r[0]:.3f} (실제 Opinion 중 찾아낸 비율)")
    print(f"   F1 Score:  {f1[0]:.3f}")
    
    # Opinion을 다른 클래스로 오분류한 경우 (Recall 문제)
    if opinion_actual_mask.sum() > 0:
        opinion_samples_preds = val_preds_np[opinion_actual_mask]
        misclassified = opinion_samples_preds != opinion_class_id
        
        print(f"\n❌ Opinion을 놓친 경우: {misclassified.sum()}/{opinion_actual_mask.sum()}개")
        
        if misclassified.sum() > 0:
            wrong_preds = opinion_samples_preds[misclassified]
            wrong_counts = Counter(wrong_preds)
            print(f"   어떤 클래스로 오분류되었나:")
            for class_id, count in wrong_counts.most_common(5):
                percentage = 100 * count / misclassified.sum()
                print(f"   → Class {class_id}: {count}번 ({percentage:.1f}%)")
    
    # 다른 클래스를 Opinion으로 오분류한 경우 (Precision 문제)
    false_positives = (val_preds_np == opinion_class_id) & (val_labels_np != opinion_class_id)
    
    if false_positives.sum() > 0:
        print(f"\n⚠️  다른 클래스를 Opinion으로 오판: {false_positives.sum()}개")
        false_positive_labels = val_labels_np[false_positives]
        fp_counts = Counter(false_positive_labels)
        print(f"   어떤 클래스가 Opinion으로 오판되었나:")
        for class_id, count in fp_counts.most_common(5):
            percentage = 100 * count / false_positives.sum()
            print(f"   → Class {class_id}: {count}번 ({percentage:.1f}%)")
    
    # 진단 및 권장사항
    print(f"\n💡 진단 및 권장사항:")
    
    if r[0] < 0.5:
        print(f"   🔴 Recall({r[0]:.3f})이 매우 낮음 → Opinion을 못 찾고 있음")
        print(f"      → Class Weight 강화 또는 Oversampling 필요")
        print(f"      → Config: MANUAL_CLASS_WEIGHTS = {{{opinion_class_id}: 3.0}}")
    elif r[0] < 0.7:
        print(f"   🟡 Recall({r[0]:.3f})이 낮은 편 → Opinion 검출 개선 필요")
        print(f"      → Class Weight 적용 권장")
    
    if p[0] < 0.5:
        print(f"   🔴 Precision({p[0]:.3f})이 매우 낮음 → 너무 많이 Opinion으로 예측")
        print(f"      → 모델 capacity 증가 또는 혼동되는 클래스와의 구분 학습")
    elif p[0] < 0.7:
        print(f"   🟡 Precision({p[0]:.3f})이 낮은 편 → 오판 줄이기 필요")
    
    if r[0] >= 0.7 and p[0] >= 0.7:
        print(f"   🟢 전반적으로 양호 → 미세 조정으로 개선 가능")
    
    print("="*70)


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
    
    # Statement of Opinion 클래스 ID (config에서 가져오거나 기본값)
    opinion_class_id = getattr(config, 'OPINION_CLASS_ID', None)
    
    print("=" * 70)
    print("🚀 K-Fold Cross Validation 시작")
    print(f"Model: {model_name}, Epochs: {epochs}, Batch: {batch_size}, Folds: {n_folds}")
    if opinion_class_id is not None:
        print(f"Statement of Opinion 클래스 ID: {opinion_class_id}")
    print("=" * 70)
    
    # ✅ Augmentation 수정!
    train_transform = get_train_augmentation(config.IMAGE_SIZE, config)
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

        # Class-Balanced Sampling 설정
        use_balanced_sampling = getattr(config, 'USE_CLASS_BALANCED_SAMPLING', False)
        sampler = None
        shuffle = True

        if use_balanced_sampling:
            # 현재 fold의 train 레이블 추출
            train_labels_fold = [train_labels[i] for i in train_idx]

            # 클래스별 샘플 수 계산
            class_counts = Counter(train_labels_fold)

            # 각 클래스의 가중치 계산 (inverse frequency)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

            # 각 샘플의 가중치 계산
            sample_weights = [class_weights[label] for label in train_labels_fold]

            # WeightedRandomSampler 생성
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True  # 복원 추출 (oversampling 효과)
            )
            shuffle = False  # sampler 사용 시 shuffle은 False

            if fold == 0:  # 첫 번째 fold에서만 출력
                print(f"\n✅ Class-Balanced Sampling 활성화")
                print(f"   샘플별 가중치로 균형잡힌 샘플링 수행")
                # 가장 적은 클래스와 가장 많은 클래스 비교
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
                min_class_id, min_count = sorted_classes[0]
                max_class_id, max_count = sorted_classes[-1]

                min_weight = class_weights[min_class_id]
                max_weight = class_weights[max_class_id]
                sampling_ratio = min_weight / max_weight  # 최소 클래스가 최대 클래스 대비 몇 배 더 자주 샘플링되는지

                print(f"   클래스별 샘플링 가중치:")
                print(f"     최소 클래스 {min_class_id}: {min_count}개 → weight {min_weight:.4f}")
                print(f"     최대 클래스 {max_class_id}: {max_count}개 → weight {max_weight:.4f}")
                print(f"     → 최소 클래스가 최대 클래스보다 {sampling_ratio:.1f}배 더 자주 샘플링됨")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            collate_fn=safe_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=safe_collate_fn
        )
        
        print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

        # 모델 생성 (매 fold마다 새로 생성)
        dropout_rate = getattr(config, 'DROPOUT_RATE', 0.0)
        model = get_model(
            model_name,
            num_classes,
            pretrained=True,
            dropout_rate=dropout_rate
        ).to(device)
        optimizer = get_optimizer(model, config)
        
        # Class weights 계산 (클래스 불균형 대응)
        use_class_weights = getattr(config, 'USE_CLASS_WEIGHTS', False)
        class_weights = None

        if use_class_weights:
            # 현재 fold의 train 레이블로 가중치 계산
            train_labels_fold = [train_labels[i] for i in train_idx]
            class_counts = Counter(train_labels_fold)

            # Power 파라미터로 가중치 조절
            power = getattr(config, 'CLASS_WEIGHT_POWER', 0.5)
            
            # Inverse frequency weighting with power
            total_samples = len(train_labels_fold)
            weights = []
            for class_id in range(num_classes):
                count = class_counts.get(class_id, 1)  # 0 방지
                weight = total_samples / (num_classes * count)
                # Power로 완화 (1.0이면 원래대로, 0.5면 제곱근)
                weight = weight ** power
                weights.append(weight)
            
            # 수동 weight 오버라이드
            manual_weights = getattr(config, 'MANUAL_CLASS_WEIGHTS', {})
            if manual_weights:
                for class_id, multiplier in manual_weights.items():
                    weights[class_id] *= multiplier

            class_weights = torch.FloatTensor(weights).to(device)

            if fold == 0:
                print(f"\n✅ Class Weights 적용 (power={power}):")
                # 샘플 수가 적은 상위 5개 클래스 표시
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
                for class_id, count in sorted_classes[:5]:
                    print(f"   Class {class_id}: {count}개 → weight {weights[class_id]:.3f}")
                print(f"   Min weight: {min(weights):.3f}, Max weight: {max(weights):.3f}")
                
                if manual_weights:
                    print(f"\n🎯 Manual weight override:")
                    for class_id, mult in manual_weights.items():
                        print(f"   Class {class_id}: ×{mult} → {weights[class_id]:.3f}")

        # Loss 함수 선택
        if config.USE_LABEL_SMOOTHING:
            criterion = LabelSmoothingLoss(
                num_classes=num_classes,
                smoothing=config.LABEL_SMOOTHING_FACTOR
            )
            if fold == 0:  # 첫 번째 fold에서만 출력
                print(f"\n✅ Label Smoothing 사용 (smoothing={config.LABEL_SMOOTHING_FACTOR})")
        else:
            # Class weights 적용
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()

            if fold == 0:
                weight_msg = " with Class Weights" if class_weights is not None else ""
                print(f"\n✅ CrossEntropyLoss 사용{weight_msg}")
        
        # Scheduler & Early Stopping
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        early_stopping_delta = getattr(config, 'EARLY_STOPPING_DELTA', 0.001)
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=early_stopping_delta)
        
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
            val_loss, val_acc, val_f1, val_preds, val_labels = validate(
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
            
            # Statement of Opinion 상세 분석 (첫 fold, 마지막 epoch 또는 early stop 시)
            if opinion_class_id is not None and fold == 0:
                if epoch == epochs - 1 or (epoch > 0 and early_stopping.early_stop):
                    analyze_opinion_class(val_labels, val_preds, opinion_class_id, num_classes)
            
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

    # ============================================
    # Save Best Fold Model (if enabled)
    # ============================================
    save_model = getattr(config, 'SAVE_MODEL', False)

    if save_model:
        # Find best fold
        best_fold_idx = np.argmax([r['best_val_f1'] for r in fold_results])
        best_fold = fold_results[best_fold_idx]
        best_f1 = best_fold['best_val_f1']

        print("\n" + "=" * 70)
        print("💾 Saving Best Model")
        print("=" * 70)
        print(f"Best Fold: {best_fold['fold']} (Val F1: {best_f1:.4f})")

        # Create models directory
        from pathlib import Path
        models_dir = Path(getattr(config, 'MODELS_DIR', 'models'))
        models_dir.mkdir(exist_ok=True)

        # Save model with descriptive filename
        model_filename = f"{model_name}_best_f1_{best_f1:.4f}.pth"
        model_path = models_dir / model_filename

        torch.save(best_fold['best_model_state'], model_path)

        print(f"✅ Model saved: {model_path}")
        print("=" * 70)
    else:
        print("\n⏭️  Model saving disabled (SAVE_MODEL=False)")

    return fold_results