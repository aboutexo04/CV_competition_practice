"""
모델 평가 및 결과 분석 모듈
"""
import ast
import json
import platform

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

_HISTORY_KEY_ALIASES = {
    'train_loss': ['train_loss', 'training_loss', 'train_losses', 'loss_train', 'trainLoss', 'training_losses'],
    'val_loss': ['val_loss', 'validation_loss', 'valid_loss', 'val_losses', 'loss_val', 'validation_losses', 'valLoss'],
    'train_acc': ['train_acc', 'train_accuracy', 'training_accuracy', 'accuracy_train', 'trainAcc', 'train_accs'],
    'val_acc': ['val_acc', 'val_accuracy', 'validation_accuracy', 'valid_acc', 'accuracy_val', 'valAcc'],
    'train_f1': ['train_f1', 'train_f1_score', 'training_f1', 'f1_train', 'train_macro_f1', 'train_f1_macro'],
    'val_f1': ['val_f1', 'val_f1_score', 'validation_f1', 'f1_val', 'val_macro_f1', 'val_f1_macro']
}

_HISTORY_FIELD_CANDIDATES = (
    'history',
    'history_json',
    'history_dict',
    'history_data',
    'metrics_history',
    'log_history'
)


def _maybe_parse_literal(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def _coerce_history_blob(blob):
    if blob is None:
        return None
    if isinstance(blob, str):
        parsed = _maybe_parse_literal(blob)
        if parsed is None:
            return None
        return _coerce_history_blob(parsed)
    if isinstance(blob, dict):
        flattened = {}
        nested_found = False
        for key, value in blob.items():
            if isinstance(value, dict):
                nested_found = True
                for sub_key, sub_value in value.items():
                    flattened[f'{key}_{sub_key}'] = sub_value
                    flattened[f'{sub_key}_{key}'] = sub_value
            else:
                flattened[key] = value
        return flattened if nested_found else blob
    if isinstance(blob, (list, tuple)):
        merged = {}
        for entry in blob:
            if not isinstance(entry, dict):
                continue
            for key, value in entry.items():
                merged.setdefault(key, []).append(value)
        return merged or None
    return None


def _ensure_list_sequence(value):
    if value is None:
        return []
    if isinstance(value, str):
        parsed = _maybe_parse_literal(value)
        if parsed is not None:
            return _ensure_list_sequence(parsed)
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return []
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_ensure_list_sequence(item))
        return result
    if isinstance(value, (np.floating, np.integer)):
        return [float(value)]
    if isinstance(value, (float, int)):
        return [float(value)]
    return []


def _collect_history_from_result(result):
    history_dict = None
    for field in _HISTORY_FIELD_CANDIDATES:
        if field in result:
            candidate = _coerce_history_blob(result[field])
            if candidate:
                history_dict = candidate
                break
    normalized = {}
    for canonical, aliases in _HISTORY_KEY_ALIASES.items():
        sequence = []
        if history_dict:
            for alias in aliases:
                if alias in history_dict:
                    sequence = _ensure_list_sequence(history_dict[alias])
                    if sequence:
                        break
        if not sequence:
            for alias in aliases:
                if alias in result:
                    sequence = _ensure_list_sequence(result[alias])
                    if sequence:
                        break
        normalized[canonical] = sequence
    if not any(len(seq) for seq in normalized.values()):
        return None
    max_length = max(len(seq) for seq in normalized.values() if len(seq))
    for key, seq in normalized.items():
        if len(seq) > max_length:
            normalized[key] = seq[:max_length]
    return normalized


def _plot_metric_panel(ax, history, train_key, val_key, title_suffix, ylabel, fold_label):
    plotted = False
    for key, color, label, marker in (
        (train_key, 'blue', 'Train', 'o'),
        (val_key, 'red', 'Val', 's'),
    ):
        values = history.get(key, [])
        if not values:
            continue
        epochs = list(range(1, len(values) + 1))
        if len(values) == 1:
            ax.scatter(epochs, values, color=color, s=100, label=label, marker=marker, zorder=5)
            ax.axhline(y=values[0], color=color, linestyle='--', alpha=0.5, linewidth=2)
            ax.set_xticks(epochs)
        else:
            ax.plot(epochs, values, color=color, linewidth=2, marker=marker, label=label)
        plotted = True
    if fold_label is not None:
        ax.set_title(f'Fold {fold_label} - {title_suffix}', fontsize=12, fontweight='bold')
    else:
        ax.set_title(title_suffix, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    else:
        ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', fontsize=11, transform=ax.transAxes)


def _prepare_image_for_display(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        img_np = image
    else:
        try:
            img_np = np.asarray(image)
        except Exception:
            return image
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        img_np = np.squeeze(img_np, axis=2)
    if img_np.dtype.kind == 'f':
        img_np = np.clip(img_np, 0, 1)
    return img_np

def test_time_augmentation(model, images, device, tta_transforms=None):
    """
    Test Time Augmentation 적용
    
    Args:
        model: 예측 모델
        images: 입력 이미지 배치 (B, C, H, W)
        device: 디바이스
        tta_transforms: TTA 변환 리스트 (None이면 기본 변환 사용)
            예: ['original', 'hflip', 'vflip', 'hflip_vflip', 'rotate90', 'rotate180', 'rotate270']
    
    Returns:
        averaged_probs: 앙상블된 확률 (B, num_classes)
    """
    if tta_transforms is None:
        tta_transforms = ['original', 'hflip']  # 기본: 원본 + 좌우 반전
    
    predictions = []
    
    for transform in tta_transforms:
        # 변환 적용
        if transform == 'original':
            augmented = images
        elif transform == 'hflip':
            augmented = torch.flip(images, dims=[3])  # 좌우 반전
        elif transform == 'vflip':
            augmented = torch.flip(images, dims=[2])  # 상하 반전
        elif transform == 'hflip_vflip':
            augmented = torch.flip(images, dims=[2, 3])  # 좌우+상하 반전
        elif transform == 'rotate90':
            augmented = torch.rot90(images, k=1, dims=[2, 3])
        elif transform == 'rotate180':
            augmented = torch.rot90(images, k=2, dims=[2, 3])
        elif transform == 'rotate270':
            augmented = torch.rot90(images, k=3, dims=[2, 3])
        else:
            continue
        
        # 예측
        with torch.no_grad():
            outputs = model(augmented.to(device))
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.cpu())
    
    # 앙상블 (평균)
    averaged_probs = torch.stack(predictions).mean(dim=0)
    
    return averaged_probs
def evaluate_ensemble(fold_results, test_dataset, config, use_tta=False, tta_transforms=None, top_k_folds=None):
    """
    K-Fold 모델들의 앙상블 평가 (Config 기반)

    Args:
        fold_results: K-Fold 학습 결과
        test_dataset: 테스트 데이터셋
        config: Config 객체
        use_tta: TTA 사용 여부
        tta_transforms: TTA 변환 리스트
        top_k_folds: 상위 K개 폴드만 사용 (None이면 모든 폴드 사용)

    Returns:
        test_acc, test_f1, ensemble_preds, test_labels
    """
    from src.model import get_model

    device = config.DEVICE
    batch_size = config.BATCH_SIZE
    num_classes = config.NUM_CLASSES
    model_name = config.MODEL_NAME

    # Top K folds 선택
    selected_fold_results = fold_results
    if top_k_folds is not None and top_k_folds > 0 and top_k_folds < len(fold_results):
        # Validation F1 스코어로 정렬 (내림차순)
        sorted_results = sorted(fold_results, key=lambda x: x.get('best_val_f1', 0), reverse=True)
        selected_fold_results = sorted_results[:top_k_folds]

        print(f"\n🎯 상위 {top_k_folds}개 폴드만 앙상블에 사용합니다:")
        for i, result in enumerate(selected_fold_results, 1):
            fold_num = result.get('fold', i)
            val_f1 = result.get('best_val_f1', 0)
            print(f"  {i}. Fold {fold_num}: Val F1 = {val_f1:.4f}")

        # 제외된 폴드 정보
        excluded_results = sorted_results[top_k_folds:]
        if excluded_results:
            print(f"\n⚠️  제외된 폴드 ({len(excluded_results)}개):")
            for result in excluded_results:
                fold_num = result.get('fold', '?')
                val_f1 = result.get('best_val_f1', 0)
                print(f"  Fold {fold_num}: Val F1 = {val_f1:.4f}")
        print()
    else:
        print(f"\n🔮 모든 {len(fold_results)}개 폴드를 앙상블에 사용합니다.\n")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("🔮 앙상블 예측 시작...")
    if use_tta:
        print(f"✅ TTA 활성화: {tta_transforms}")
    else:
        print("❌ TTA 비활성화")
    all_predictions = []

    for fold_idx, fold_result in enumerate(selected_fold_results):
        # 체크포인트와 같은 dropout_rate로 모델 생성 (eval 모드에서 자동 비활성화됨)
        # fold_result에 dropout_rate 정보가 있으면 사용, 없으면 0.0
        dropout_rate_for_model = fold_result.get('dropout_rate', 0.0)
        fold_model = get_model(model_name, num_classes, pretrained=False, dropout_rate=dropout_rate_for_model)
        fold_model.load_state_dict(fold_result['best_model_state'])
        fold_model = fold_model.to(device)
        fold_model.eval()

        fold_preds = []
        with torch.no_grad():
            desc = f"Fold {fold_idx + 1} {'(TTA)' if use_tta else ''}"
            for images, labels in tqdm(test_loader, desc=desc, leave=True):
                if use_tta:
                    # TTA 적용 - 텐서 반환
                    probs = test_time_augmentation(fold_model, images, device, tta_transforms)
                    fold_preds.append(probs.cpu().numpy())  # ✅ .cpu() 추가
                else:
                    # 일반 예측
                    images = images.to(device)
                    outputs = fold_model(images)
                    probs = torch.softmax(outputs, dim=1)
                    fold_preds.append(probs.cpu().numpy())
        fold_preds = np.concatenate(fold_preds, axis=0)
        all_predictions.append(fold_preds)
        tta_status = "with TTA" if use_tta else "standard"
        print(f"✅ Fold {fold_idx + 1} 예측 완료! ({tta_status}, shape: {fold_preds.shape})")  # ✅ shape 정보 추가

    # 앙상블 (평균)
    ensemble_probs = np.mean(all_predictions, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    # 실제 레이블
    test_labels = [label for _, label in test_dataset]
    
    # 레이블 확인: 모든 레이블이 -1이 아니고 유효한 범위 내에 있는지 체크
    has_true_labels = False
    if test_labels and len(test_labels) > 0:
        # -1이 아닌 유효한 레이블이 하나라도 있으면 실제 레이블이 있다고 판단
        valid_labels = [label for label in test_labels if label != -1 and label >= 0]
        has_true_labels = len(valid_labels) > 0

    if has_true_labels:
        # 평가 지표 계산 (유효한 레이블만 사용)
        valid_indices = [i for i, label in enumerate(test_labels) if label != -1 and label >= 0]
        valid_test_labels = [test_labels[i] for i in valid_indices]
        valid_ensemble_preds = [ensemble_preds[i] for i in valid_indices]
        
        test_f1 = f1_score(valid_test_labels, valid_ensemble_preds, average='macro')
        test_acc = 100. * np.sum(np.array(valid_ensemble_preds) == np.array(valid_test_labels)) / len(valid_test_labels)

        print("\n" + "=" * 70)
        print("🎯 Test Set 최종 결과 (앙상블)")
        print("=" * 70)
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Macro F1 Score: {test_f1:.4f}")
        print(f"✅ 실제 테스트 레이블로 계산된 정확한 F1 스코어입니다.")
        print("=" * 70)
    else:
        # 실제 테스트 데이터 - 레이블 없음
        # Validation F1 평균을 사용 (제출 파일명에 사용)
        if fold_results and len(fold_results) > 0:
            avg_val_f1 = np.mean([r['best_val_f1'] for r in fold_results if 'best_val_f1' in r])
            test_f1 = avg_val_f1
            print("\n" + "=" * 70)
            print("🎯 Test Set 최종 결과 (앙상블)")
            print("=" * 70)
            print("⚠️  실제 테스트 데이터는 레이블이 없어 평가할 수 없습니다.")
            print(f"✅ 예측 완료: {len(ensemble_preds):,}개 샘플")
            print(f"📊 제출 파일명에 사용할 F1 스코어 (평균 Validation F1): {test_f1:.4f}")
            print("📝 제출 파일을 생성하여 대회에 제출하세요.")
            print("=" * 70)
        else:
            # Fallback: fold_results가 없거나 비어있는 경우
            test_f1 = 0.0
            print("\n" + "=" * 70)
            print("🎯 Test Set 최종 결과 (앙상블)")
            print("=" * 70)
            print("⚠️  실제 테스트 데이터는 레이블이 없고, Validation F1도 계산할 수 없습니다.")
            print(f"✅ 예측 완료: {len(ensemble_preds):,}개 샘플")
            print(f"❌ F1 스코어를 계산할 수 없습니다. (기본값: 0.0000)")
            print("📝 제출 파일을 생성하여 대회에 제출하세요.")
            print("=" * 70)
        
        test_acc = 0.0

    return test_acc, test_f1, ensemble_preds, test_labels


def evaluate_single_model(model, test_dataset, config, use_tta=False, tta_transforms=None):
    """
    단일 모델 평가 (전체 데이터로 학습한 최종 모델용)

    Args:
        model: 학습된 모델
        test_dataset: 테스트 데이터셋
        config: Config 객체
        use_tta: TTA 사용 여부
        tta_transforms: TTA 변환 리스트

    Returns:
        test_acc, test_f1, predictions, test_labels
    """
    device = config.DEVICE
    batch_size = config.BATCH_SIZE

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("🔮 단일 모델 예측 시작...")

    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="예측 중", leave=True):
            if use_tta:
                # TTA 적용
                probs = test_time_augmentation(model, images, device, tta_transforms)
                all_preds.append(probs.cpu().numpy())
            else:
                # 일반 예측
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    predictions = np.argmax(all_preds, axis=1)

    print(f"✅ 예측 완료! (predictions shape: {all_preds.shape})")

    # 실제 레이블
    test_labels = [label for _, label in test_dataset]

    # 레이블 확인
    has_true_labels = False
    if test_labels and len(test_labels) > 0:
        valid_labels = [label for label in test_labels if label != -1 and label >= 0]
        has_true_labels = len(valid_labels) > 0

    if has_true_labels:
        # 평가 지표 계산
        valid_indices = [i for i, label in enumerate(test_labels) if label != -1 and label >= 0]
        valid_test_labels = [test_labels[i] for i in valid_indices]
        valid_predictions = [predictions[i] for i in valid_indices]

        test_f1 = f1_score(valid_test_labels, valid_predictions, average='macro')
        test_acc = 100. * np.sum(np.array(valid_predictions) == np.array(valid_test_labels)) / len(valid_test_labels)

        print("\n" + "=" * 70)
        print("🎯 Test Set 최종 결과 (단일 모델)")
        print("=" * 70)
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Macro F1 Score: {test_f1:.4f}")
        print(f"✅ 실제 테스트 레이블로 계산된 정확한 F1 스코어입니다.")
        print("=" * 70)
    else:
        # 실제 테스트 데이터 - 레이블 없음
        test_f1 = 0.0
        test_acc = 0.0
        print("\n" + "=" * 70)
        print("🎯 Test Set 최종 결과 (단일 모델)")
        print("=" * 70)
        print("⚠️  실제 테스트 데이터는 레이블이 없어 평가할 수 없습니다.")
        print(f"✅ 예측 완료: {len(predictions):,}개 샘플")
        print("📝 제출 파일을 생성하여 대회에 제출하세요.")
        print("=" * 70)

    return test_acc, test_f1, predictions, test_labels


def plot_training_curves(fold_results):
    """
    K-Fold 학습 곡선 시각화

    Args:
        fold_results: K-Fold 학습 결과 리스트
    """
    if len(fold_results) == 0:
        print("⚠️ 학습 결과가 없습니다.")
        return

    normalized_results = []
    missing_history_folds = []
    for idx, result in enumerate(fold_results):
        if not isinstance(result, dict):
            missing_history_folds.append(idx + 1)
            continue
        history = _collect_history_from_result(result)
        if history:
            fold_label = result.get('fold', idx + 1)
            normalized_results.append((fold_label, history))
        else:
            missing_history_folds.append(result.get('fold', idx + 1))

    if not normalized_results:
        print("⚠️ 학습 history가 포함된 fold가 없습니다. 시각화를 건너뜁니다.")
        return

    if len(normalized_results) < len(fold_results):
        unique_missing = [f for f in missing_history_folds if f is not None]
        if unique_missing:
            missing_text = ', '.join(str(f) for f in sorted(set(unique_missing)))
            print(f"ℹ️ history를 찾지 못한 fold: {missing_text}")
        print(f"ℹ️ 일부 fold에 history가 없어 {len(normalized_results)}/{len(fold_results)} fold만 시각화합니다.")

    num_folds = len(normalized_results)
    fig, axes = plt.subplots(num_folds, 3, figsize=(18, 5 * num_folds))
    axes = np.atleast_2d(axes)

    for row_idx, (fold_label, history) in enumerate(normalized_results):
        display_label = fold_label if fold_label is not None else row_idx + 1
        _plot_metric_panel(axes[row_idx, 0], history, 'train_loss', 'val_loss', 'Loss', 'Loss', display_label)
        _plot_metric_panel(axes[row_idx, 1], history, 'train_acc', 'val_acc', 'Accuracy', 'Accuracy (%)', display_label)
        _plot_metric_panel(axes[row_idx, 2], history, 'train_f1', 'val_f1', 'F1 Score', 'F1 Score', display_label)

    plt.tight_layout()
    plt.show()
    print("✅ 학습 곡선 시각화 완료!")


def plot_confusion_matrix(test_labels, predictions, class_names):
    """
    Confusion Matrix 시각화

    Args:
        test_labels: 실제 레이블
        predictions: 예측 레이블
        class_names: 클래스 이름 리스트
    """
    cm = confusion_matrix(test_labels, predictions)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (앙상블)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 클래스별 정확도
    print("\n" + "=" * 70)
    print("📊 클래스별 정확도")
    print("=" * 70)
    for i in range(len(class_names)):
        true_positives = cm[i, i]
        total = cm[i, :].sum()
        acc = (true_positives / total * 100) if total > 0 else 0
        print(f"{class_names[i]:20s}: {acc:5.2f}% ({true_positives}/{total})")
    print("=" * 70)


def analyze_misclassifications(test_dataset_raw, test_labels, predictions, class_names, max_samples=16, fallback_dataset=None):
    """
    오분류 분석 및 시각화

    Args:
        test_dataset_raw: 원본 테스트 데이터셋 (transform 없이)
        test_labels: 실제 레이블
        predictions: 예측 레이블
        class_names: 클래스 이름 리스트
        max_samples: 최대 표시 샘플 수
        fallback_dataset: 원본 데이터가 없을 때 사용할 대체 데이터셋
    """
    test_labels_array = np.array(test_labels)
    predictions_array = np.array(predictions)

    # 오분류 찾기
    wrong_predictions = test_labels_array != predictions_array
    wrong_indices = np.where(wrong_predictions)[0]

    print("\n" + "=" * 70)
    print("📊 오분류 분석 결과")
    print("=" * 70)
    print(f"전체 테스트 샘플: {len(test_labels_array):,}개")
    print(f"정확히 예측: {np.sum(~wrong_predictions):,}개 ({100 * np.sum(~wrong_predictions) / len(test_labels_array):.2f}%)")
    print(f"오분류: {len(wrong_indices):,}개 ({100 * len(wrong_indices) / len(test_labels_array):.2f}%)")
    print("=" * 70)

    if len(wrong_indices) == 0:
        print("\n✅ 모든 샘플을 정확히 예측했습니다!")
        return

    dataset_for_visual = test_dataset_raw or fallback_dataset
    if test_dataset_raw is None and fallback_dataset is not None:
        print("\nℹ️ 원본 테스트 이미지가 없어 transform이 적용된 데이터셋으로 시각화를 진행합니다.")

    # 오분류 패턴 분석
    print("\n📈 가장 많이 틀린 클래스 조합 (Top 10):")
    print("-" * 70)

    misclassification_patterns = {}
    for idx in wrong_indices:
        true_label = test_labels_array[idx]
        pred_label = predictions_array[idx]
        pattern = (true_label, pred_label)
        misclassification_patterns[pattern] = misclassification_patterns.get(pattern, 0) + 1

    top_mistakes = sorted(misclassification_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    for (true_idx, pred_idx), count in top_mistakes:
        print(f"{class_names[true_idx]:20s} → {class_names[pred_idx]:20s}: {count:3d}회")

    # 오분류 샘플 시각화
    if dataset_for_visual is None:
        print("\n⚠️ 오분류 샘플 시각화를 위한 데이터셋이 없어 이미지를 표시하지 않습니다.")
    else:
        n_samples = min(max_samples, len(wrong_indices))
        rng = np.random.RandomState()
        selected_wrong_indices = rng.choice(wrong_indices, n_samples, replace=False)

        fig_cols = min(4, max(1, int(np.ceil(np.sqrt(n_samples)))))
        fig_rows = int(np.ceil(n_samples / fig_cols))

        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows))
        if isinstance(axes, np.ndarray):
            axes_array = axes.astype(object)
            if axes_array.ndim == 0:
                axes_array = axes_array.reshape(1, 1)
            elif axes_array.ndim == 1:
                if fig_rows == 1:
                    axes_array = axes_array.reshape(1, -1)
                else:
                    axes_array = axes_array.reshape(-1, 1)
        else:
            axes_array = np.array([[axes]], dtype=object)

        fig.suptitle('오분류된 샘플 이미지', fontsize=16, fontweight='bold', y=0.995)

        for idx, wrong_idx in enumerate(selected_wrong_indices):
            row = idx // fig_cols
            col = idx % fig_cols

            sample = dataset_for_visual[wrong_idx]
            if isinstance(sample, dict):
                image = sample.get('image')
                if image is None:
                    image = sample.get('img', sample)
            elif isinstance(sample, (tuple, list)):
                image = sample[0]
            else:
                image = sample

            image_to_show = _prepare_image_for_display(image)
            true_label = test_labels_array[wrong_idx]
            pred_label = predictions_array[wrong_idx]

            ax = axes_array[row, col]
            ax.imshow(image_to_show)
            ax.set_title(
                f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                fontsize=9,
                fontweight='bold'
            )
            ax.axis('off')

        total_slots = fig_rows * fig_cols
        for idx in range(n_samples, total_slots):
            row = idx // fig_cols
            col = idx % fig_cols
            axes_array[row, col].axis('off')

        plt.tight_layout()
        plt.show()
        print(f"\n✅ {n_samples}개 오분류 샘플 시각화 완료!")

    # 클래스별 오분류 비율 시각화
    print("\n📊 클래스별 오분류 비율 시각화...")
    class_error_rates = []
    for i in range(len(class_names)):
        errors = np.sum((test_labels_array == i) & (predictions_array != i))
        total = np.sum(test_labels_array == i)
        error_rate = (errors / total * 100) if total > 0 else 0
        class_error_rates.append(error_rate)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['red' if rate > np.mean(class_error_rates) else 'steelblue' for rate in class_error_rates]
    bars = ax.bar(range(len(class_names)), class_error_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('클래스별 오분류 비율', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 평균선 추가
    avg_error_rate = np.mean(class_error_rates)
    ax.axhline(y=avg_error_rate, color='red', linestyle='--', linewidth=2, label=f'평균: {avg_error_rate:.1f}%')
    ax.legend()

    # 값 표시
    for i, (bar, rate) in enumerate(zip(bars, class_error_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    print("✅ 클래스별 오분류 비율 시각화 완료!")


def run_full_evaluation(fold_results, test_dataset, class_names, config, train_dataset_raw=None, train_labels=None):
    """
    전체 평가 프로세스를 한 번에 실행 (Config 기반)

    Args:
        fold_results: K-Fold 학습 결과
        test_dataset: 테스트 데이터셋
        class_names: 클래스 이름 리스트
        config: Config 객체
        train_dataset_raw: 학습 데이터셋 (optional)
        train_labels: 학습 레이블 (optional)

    Returns:
        results: 평가 결과 딕셔너리
    """
    from src.data import load_data, get_val_augmentation
    from src.train import TransformSubset
    from src.model import get_model
    from sklearn.model_selection import StratifiedKFold

    device = config.DEVICE
    use_wandb = config.USE_WANDB

    print("=" * 70)
    print("🚀 전체 평가 프로세스 시작")
    print("=" * 70)

    # 1. 학습 곡선 시각화 (비활성화)
    # print("\n📈 학습 곡선 시각화...")
    # plot_training_curves(fold_results)

    # 2. Validation 데이터로 앙상블 평가
    print("\n🔮 Validation 앙상블 예측 시작...")

    # train_dataset_raw와 train_labels 가져오기
    if train_dataset_raw is None or train_labels is None:
        print("⚠️  train_dataset_raw 또는 train_labels가 없어 데이터를 다시 로드합니다...")
        train_dataset_raw, _, train_labels, _, _ = load_data(config)

    val_transform = get_val_augmentation(config.IMAGE_SIZE)
    kfold = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)

    all_val_preds = []
    all_val_labels = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_dataset_raw, train_labels)):
        print(f"\n📁 Fold {fold_idx + 1} Validation 예측...")

        # Validation 데이터셋
        val_subset = Subset(train_dataset_raw, val_idx)
        val_dataset = TransformSubset(val_subset, val_transform)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        # 모델 로드 (추론 시 dropout 비활성화)
        fold_model = get_model(config.MODEL_NAME, config.NUM_CLASSES, pretrained=False, dropout_rate=0.0)
        fold_model.load_state_dict(fold_results[fold_idx]['best_model_state'])
        fold_model = fold_model.to(device)
        fold_model.eval()

        # 예측
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Fold {fold_idx + 1}", leave=True):
                images = images.to(device)
                outputs = fold_model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(labels.numpy())

        all_val_preds.extend(fold_preds)
        all_val_labels.extend(fold_labels)
        print(f"✅ Fold {fold_idx + 1} Validation 예측 완료!")

    # Validation 성능 계산
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_acc = 100. * np.sum(np.array(all_val_preds) == np.array(all_val_labels)) / len(all_val_labels)

    print("\n" + "=" * 70)
    print("🎯 Validation Set 앙상블 결과")
    print("=" * 70)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Validation Macro F1 Score: {val_f1:.4f}")
    print("=" * 70)

    # 3. Confusion Matrix (비활성화)
    # print("\n📊 Confusion Matrix 생성...")
    # plot_confusion_matrix(all_val_labels, all_val_preds, class_names)

    # 4. 오분류 분석 (비활성화)
    # print("\n🔍 오분류 분석...")
    # analyze_misclassifications(
    #     test_dataset_raw=train_dataset_raw,
    #     test_labels=all_val_labels,
    #     predictions=all_val_preds,
    #     class_names=class_names,
    #     fallback_dataset=None
    # )

    # 5. Test 데이터 예측 (제출용)
    print("\n" + "=" * 70)
    print("🔮 Test 데이터 예측 (제출용)")
    print("=" * 70)

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    all_test_preds = []

    for fold_idx, fold_result in enumerate(fold_results):
        # 추론 시 dropout 비활성화
        fold_model = get_model(config.MODEL_NAME, config.NUM_CLASSES, pretrained=False, dropout_rate=0.0)
        fold_model.load_state_dict(fold_result['best_model_state'])
        fold_model = fold_model.to(device)
        fold_model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"Fold {fold_idx + 1} 예측", leave=True):
                images = images.to(device)
                outputs = fold_model(images)
                probs = torch.softmax(outputs, dim=1)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        all_test_preds.append(fold_preds)
        print(f"✅ Fold {fold_idx + 1} Test 예측 완료!")

    # 앙상블 (평균)
    ensemble_probs = np.mean(all_test_preds, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    print("\n" + "=" * 70)
    print(f"✅ Test 예측 완료: {len(ensemble_preds):,}개 샘플")
    print("=" * 70)

    # Wandb 로깅
    if use_wandb:
        import wandb
        wandb.log({
            "ensemble/val_accuracy": val_acc,
            "ensemble/val_f1_macro": val_f1
        })

    print("\n" + "=" * 70)
    print("✅ 전체 평가 완료!")
    print("=" * 70)

    return {
        'test_acc': val_acc,  # Validation 성능
        'test_f1': val_f1,    # Validation 성능
        'predictions': ensemble_preds,  # Test 예측값
        'labels': [-1] * len(ensemble_preds)
    }