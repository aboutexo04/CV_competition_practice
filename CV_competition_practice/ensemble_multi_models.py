#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Ensemble Script

Ensembles predictions from different model architectures (e.g., M and S models).
This script loads checkpoints from different models and averages their predictions.

Usage:
    # Ensemble M and S models (default: 2 M models + 2 S models)
    python ensemble_multi_models.py

    # Custom configuration
    python ensemble_multi_models.py \
        --m_checkpoint_pattern="*tf_efficientnetv2_m*_best_f1_*.pth" \
        --s_checkpoint_pattern="*tf_efficientnetv2_s*_best_f1_*.pth" \
        --m_n_folds=2 \
        --s_n_folds=2 \
        --m_dropout_rate=0.4 \
        --s_dropout_rate=0.3

    # With TTA
    python ensemble_multi_models.py --use_tta=True

    # Custom weights for each model type
    python ensemble_multi_models.py --m_weight=0.6 --s_weight=0.4
"""

import sys
from pathlib import Path
import torch
import numpy as np
import warnings
import fire
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================
# Add project root to path
# ============================================
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================
# Import modules
# ============================================
from src.config import DocumentConfig
from src.data import load_data
from src.submission import save_submission
from src.utils import set_seed
from src.model import get_model
from torch.utils.data import DataLoader


def ensemble_multi_models(
    # M model settings
    m_model_name='tf_efficientnetv2_m',
    m_checkpoint_pattern="*tf_efficientnetv2_m*_best_f1_*.pth",
    m_n_folds=2,
    m_dropout_rate=0.4,
    m_weight=1.0,

    # S model settings
    s_model_name='tf_efficientnetv2_s',
    s_checkpoint_pattern="*tf_efficientnetv2_s*_best_f1_*.pth",
    s_n_folds=2,
    s_dropout_rate=0.3,
    s_weight=1.0,

    # General settings
    use_tta=False,
    batch_size=32,
    seed=42,
):
    """
    Ensemble predictions from multiple model architectures

    Args:
        m_model_name: M model architecture name
        m_checkpoint_pattern: Pattern to find M model checkpoints
        m_n_folds: Number of M model checkpoints to load
        m_dropout_rate: Dropout rate used in M model training
        m_weight: Weight for M model predictions (default: 1.0)

        s_model_name: S model architecture name
        s_checkpoint_pattern: Pattern to find S model checkpoints
        s_n_folds: Number of S model checkpoints to load
        s_dropout_rate: Dropout rate used in S model training
        s_weight: Weight for S model predictions (default: 1.0)

        use_tta: Use Test Time Augmentation
        batch_size: Batch size for inference
        seed: Random seed
    """

    print("\n" + "="*70)
    print("Multi-Model Ensemble Script")
    print("="*70)

    # Set seed
    set_seed(seed)

    # ============================================
    # 1. Load Data
    # ============================================
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)

    config = DocumentConfig()
    config.BATCH_SIZE = batch_size
    device = config.DEVICE

    train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_data(config)

    if test_dataset is None:
        print("\n❌ Error: No test data found!")
        return

    print(f"\nTest samples: {len(test_dataset):,}")
    print(f"Classes ({num_classes}): {class_names}")

    # ============================================
    # 2. Load M Model Checkpoints
    # ============================================
    print("\n" + "="*70)
    print(f"Loading {m_model_name} Checkpoints")
    print("="*70)

    models_dir = Path('models')
    m_checkpoint_files = sorted(list(models_dir.glob(m_checkpoint_pattern)))
    m_checkpoint_files = m_checkpoint_files[:m_n_folds] if len(m_checkpoint_files) >= m_n_folds else m_checkpoint_files

    if not m_checkpoint_files:
        print(f"\n⚠️  Warning: No {m_model_name} checkpoints found")
        print(f"   Pattern: {m_checkpoint_pattern}")
        m_models = []
        m_f1_scores = []
    else:
        print(f"\nFound {len(m_checkpoint_files)} {m_model_name} checkpoints:")
        for i, ckpt in enumerate(m_checkpoint_files, 1):
            print(f"  {i}. {ckpt.name}")

        m_models = []
        m_f1_scores = []

        for i, checkpoint_file in enumerate(m_checkpoint_files):
            print(f"\n  Loading {m_model_name} {i+1}/{len(m_checkpoint_files)}...")

            try:
                # Create model
                model = get_model(m_model_name, num_classes=num_classes, pretrained=False, dropout_rate=m_dropout_rate)

                # Load checkpoint
                state_dict = torch.load(checkpoint_file, map_location=device)
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys or unexpected_keys:
                    print(f"    ⚠️  Warning: Model structure mismatch")
                    if missing_keys:
                        print(f"       Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"       Unexpected keys: {len(unexpected_keys)}")

                model = model.to(device)
                model.eval()

                # Extract F1 score from filename
                import re
                match = re.search(r'f1_([\d.]+)', checkpoint_file.stem, re.IGNORECASE)
                val_f1 = float(match.group(1)) if match else 0.0

                m_models.append(model)
                m_f1_scores.append(val_f1)

                print(f"    ✅ Loaded: {checkpoint_file.name} (Val F1: {val_f1:.4f})")

            except Exception as e:
                print(f"    ❌ Error loading {checkpoint_file.name}: {e}")
                continue

        print(f"\n✅ Successfully loaded {len(m_models)} {m_model_name} models")
        if m_f1_scores:
            print(f"   Average F1: {np.mean(m_f1_scores):.4f}")

    # ============================================
    # 3. Load S Model Checkpoints
    # ============================================
    print("\n" + "="*70)
    print(f"Loading {s_model_name} Checkpoints")
    print("="*70)

    s_checkpoint_files = sorted(list(models_dir.glob(s_checkpoint_pattern)))
    s_checkpoint_files = s_checkpoint_files[:s_n_folds] if len(s_checkpoint_files) >= s_n_folds else s_checkpoint_files

    if not s_checkpoint_files:
        print(f"\n⚠️  Warning: No {s_model_name} checkpoints found")
        print(f"   Pattern: {s_checkpoint_pattern}")
        s_models = []
        s_f1_scores = []
    else:
        print(f"\nFound {len(s_checkpoint_files)} {s_model_name} checkpoints:")
        for i, ckpt in enumerate(s_checkpoint_files, 1):
            print(f"  {i}. {ckpt.name}")

        s_models = []
        s_f1_scores = []

        for i, checkpoint_file in enumerate(s_checkpoint_files):
            print(f"\n  Loading {s_model_name} {i+1}/{len(s_checkpoint_files)}...")

            try:
                # Create model
                model = get_model(s_model_name, num_classes=num_classes, pretrained=False, dropout_rate=s_dropout_rate)

                # Load checkpoint
                state_dict = torch.load(checkpoint_file, map_location=device)
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys or unexpected_keys:
                    print(f"    ⚠️  Warning: Model structure mismatch")
                    if missing_keys:
                        print(f"       Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"       Unexpected keys: {len(unexpected_keys)}")

                model = model.to(device)
                model.eval()

                # Extract F1 score from filename
                import re
                match = re.search(r'f1_([\d.]+)', checkpoint_file.stem, re.IGNORECASE)
                val_f1 = float(match.group(1)) if match else 0.0

                s_models.append(model)
                s_f1_scores.append(val_f1)

                print(f"    ✅ Loaded: {checkpoint_file.name} (Val F1: {val_f1:.4f})")

            except Exception as e:
                print(f"    ❌ Error loading {checkpoint_file.name}: {e}")
                continue

        print(f"\n✅ Successfully loaded {len(s_models)} {s_model_name} models")
        if s_f1_scores:
            print(f"   Average F1: {np.mean(s_f1_scores):.4f}")

    # Check if any models loaded
    total_models = len(m_models) + len(s_models)
    if total_models == 0:
        print("\n❌ Error: No models loaded successfully!")
        return

    # ============================================
    # 4. Generate Predictions
    # ============================================
    print("\n" + "="*70)
    print("Generating Ensemble Predictions")
    print("="*70)

    print(f"\nTotal models: {total_models}")
    print(f"  {m_model_name}: {len(m_models)} models (weight: {m_weight:.2f})")
    print(f"  {s_model_name}: {len(s_models)} models (weight: {s_weight:.2f})")
    print(f"TTA: {'Yes' if use_tta else 'No'}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # TTA transforms
    if use_tta:
        tta_transforms_list = ['original', 'hflip', 'vflip', 'rotate90']
        print(f"TTA transforms: {tta_transforms_list}")
    else:
        tta_transforms_list = ['original']

    # Collect all predictions
    all_predictions = []

    # M model predictions
    if m_models:
        print(f"\n{'='*50}")
        print(f"Generating {m_model_name} predictions...")
        print(f"{'='*50}")

        for model_idx, model in enumerate(m_models):
            print(f"\n  Model {model_idx+1}/{len(m_models)}...")

            for tta_name in tta_transforms_list:
                model_predictions = []

                with torch.no_grad():
                    for images, _ in tqdm(test_loader, desc=f"    TTA: {tta_name}"):
                        images = images.to(device)

                        # Apply TTA
                        if tta_name == 'hflip':
                            images = torch.flip(images, dims=[3])
                        elif tta_name == 'vflip':
                            images = torch.flip(images, dims=[2])
                        elif tta_name == 'rotate90':
                            images = torch.rot90(images, k=1, dims=[2, 3])

                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1)
                        model_predictions.append(probs.cpu().numpy())

                model_predictions = np.concatenate(model_predictions, axis=0)
                all_predictions.append(model_predictions * m_weight)

    # S model predictions
    if s_models:
        print(f"\n{'='*50}")
        print(f"Generating {s_model_name} predictions...")
        print(f"{'='*50}")

        for model_idx, model in enumerate(s_models):
            print(f"\n  Model {model_idx+1}/{len(s_models)}...")

            for tta_name in tta_transforms_list:
                model_predictions = []

                with torch.no_grad():
                    for images, _ in tqdm(test_loader, desc=f"    TTA: {tta_name}"):
                        images = images.to(device)

                        # Apply TTA
                        if tta_name == 'hflip':
                            images = torch.flip(images, dims=[3])
                        elif tta_name == 'vflip':
                            images = torch.flip(images, dims=[2])
                        elif tta_name == 'rotate90':
                            images = torch.rot90(images, k=1, dims=[2, 3])

                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1)
                        model_predictions.append(probs.cpu().numpy())

                model_predictions = np.concatenate(model_predictions, axis=0)
                all_predictions.append(model_predictions * s_weight)

    # Average all predictions
    print(f"\n{'='*50}")
    print("Averaging predictions...")
    print(f"{'='*50}")

    ensemble_probs = np.mean(all_predictions, axis=0)
    final_predictions = np.argmax(ensemble_probs, axis=1)

    print(f"\nFinal predictions shape: {final_predictions.shape}")
    print(f"Unique predictions: {len(np.unique(final_predictions))} classes")

    # ============================================
    # 5. Save Submission
    # ============================================
    print("\n" + "="*70)
    print("Saving Submission File")
    print("="*70)

    submission_dir = Path('submissions')
    submission_dir.mkdir(exist_ok=True)

    # Calculate average F1 for filename
    all_f1_scores = m_f1_scores + s_f1_scores
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0

    suffix = f'multi_M{len(m_models)}_S{len(s_models)}'
    if use_tta:
        suffix += '_tta'

    save_submission(
        preds=final_predictions,
        sample_path=config.SUBMISSION_PATH,
        save_path=submission_dir,
        f1_score=avg_f1,
        suffix=suffix
    )

    # ============================================
    # 6. Summary
    # ============================================
    print("\n" + "="*70)
    print("Multi-Model Ensemble Complete!")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  {m_model_name}: {len(m_models)} models (weight: {m_weight:.2f})")
    if m_f1_scores:
        print(f"    F1 scores: {[f'{f:.4f}' for f in m_f1_scores]}")
        print(f"    Average: {np.mean(m_f1_scores):.4f}")

    print(f"  {s_model_name}: {len(s_models)} models (weight: {s_weight:.2f})")
    if s_f1_scores:
        print(f"    F1 scores: {[f'{f:.4f}' for f in s_f1_scores]}")
        print(f"    Average: {np.mean(s_f1_scores):.4f}")

    print(f"\n  Total models: {total_models}")
    print(f"  TTA: {'Yes' if use_tta else 'No'}")
    print(f"  Total predictions: {len(final_predictions):,}")
    print(f"  Average CV F1: {avg_f1:.4f}")

    print("\nClass Distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(final_predictions == i)
        percentage = count / len(final_predictions) * 100
        print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

    print("="*70)


if __name__ == "__main__":
    try:
        fire.Fire(ensemble_multi_models)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
