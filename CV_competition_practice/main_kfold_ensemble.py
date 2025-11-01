#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Fold Ensemble Submission Script

Uses saved K-Fold checkpoint files to create ensemble predictions.
This script loads fold-specific checkpoints (NOT final models).

IMPORTANT: This loads fold checkpoints like *_best_f1_*.pth or fold-specific files.
           Each fold model was trained on 80% of data.
           For better performance, use final models with main_ensemble.py instead.

Usage:
    # Use saved fold_results.pkl
    python submit_kfold_ensemble.py

    # Specify model name and number of folds
    python submit_kfold_ensemble.py --model_name=tf_efficientnetv2_m --n_folds=5

    # With TTA
    python submit_kfold_ensemble.py --use_tta=True
"""

import sys
from pathlib import Path
import torch
import numpy as np
import warnings
import fire
import pickle
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
from src.evaluation import evaluate_ensemble
from src.submission import save_submission
from src.utils import set_seed


def submit_from_fold_results(
    use_tta=True,
    batch_size=32,
    seed=42,
):
    """
    Create submission using saved fold_results.pkl

    Args:
        use_tta: Use Test Time Augmentation
        batch_size: Batch size for inference
        seed: Random seed
    """

    print("\n" + "="*70)
    print("K-Fold Ensemble Submission (from fold_results.pkl)")
    print("="*70)

    # Set seed
    set_seed(seed)

    # ============================================
    # 1. Load fold_results.pkl
    # ============================================
    results_path = Path('results/fold_results.pkl')

    if not results_path.exists():
        print(f"\n❌ Error: fold_results.pkl not found at {results_path}")
        print("   Please run main_full.py first to train models and save fold results.")
        return

    print(f"\nLoading fold results from: {results_path}")

    with open(results_path, 'rb') as f:
        fold_results = pickle.load(f)

    print(f"✅ Loaded {len(fold_results)} fold results")

    # ============================================
    # 2. Load Config and Data
    # ============================================
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)

    config = DocumentConfig()
    config.BATCH_SIZE = batch_size

    train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_data(config)

    if test_dataset is None:
        print("\n❌ Error: No test data found!")
        return

    print(f"\nTest samples: {len(test_dataset):,}")
    print(f"Classes ({num_classes}): {class_names}")

    # ============================================
    # 3. Ensemble Prediction
    # ============================================
    print("\n" + "="*70)
    print("Running K-Fold Ensemble Prediction")
    print("="*70)

    tta_transforms = ['original', 'hflip', 'vflip', 'rotate90'] if use_tta else ['original']

    _, test_f1, predictions, _ = evaluate_ensemble(
        fold_results=fold_results,
        test_dataset=test_dataset,
        config=config,
        use_tta=use_tta,
        tta_transforms=tta_transforms
    )

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")

    # ============================================
    # 4. Save Submission
    # ============================================
    print("\n" + "="*70)
    print("Saving Submission File")
    print("="*70)

    submission_dir = Path('submissions')
    submission_dir.mkdir(exist_ok=True)

    suffix = 'kfold_tta' if use_tta else 'kfold'

    save_submission(
        preds=predictions,
        sample_path=config.SUBMISSION_PATH,
        save_path=submission_dir,
        f1_score=test_f1,
        suffix=suffix
    )

    # ============================================
    # 5. Summary
    # ============================================
    print("\n" + "="*70)
    print("K-Fold Ensemble Complete!")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Number of folds: {len(fold_results)}")
    print(f"  TTA: {'Yes' if use_tta else 'No'}")
    print(f"  Total predictions: {len(predictions):,}")
    print(f"  CV F1 Score: {test_f1:.4f}")

    # Fold summary
    print(f"\nFold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Val F1 = {result['best_val_f1']:.4f}")

    avg_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in fold_results])
    print(f"\n  Average: {avg_f1:.4f} +/- {std_f1:.4f}")

    print("\nClass Distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(predictions == i)
        percentage = count / len(predictions) * 100
        print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

    print("="*70)


def submit_from_checkpoints(
    model_name='tf_efficientnetv2_m',
    n_folds=5,
    use_tta=False,
    batch_size=32,
    seed=42,
):
    """
    Create submission by loading fold checkpoint files directly

    Args:
        model_name: Model name to load checkpoints for
        n_folds: Number of folds to load
        use_tta: Use Test Time Augmentation
        batch_size: Batch size for inference
        seed: Random seed
    """

    print("\n" + "="*70)
    print("K-Fold Ensemble Submission (from checkpoint files)")
    print("="*70)

    print(f"\n⚠️  WARNING: This method loads fold checkpoints directly.")
    print(f"   Each fold was trained on only 80% of data.")
    print(f"   For better performance, use final models with main_ensemble.py")

    # Set seed
    set_seed(seed)

    # ============================================
    # 1. Load fold checkpoints
    # ============================================
    print("\n" + "="*70)
    print("Loading Fold Checkpoints")
    print("="*70)

    models_dir = Path('models')
    checkpoint_pattern = f"*{model_name}*_best_f1_*.pth"
    checkpoint_files = sorted(list(models_dir.glob(checkpoint_pattern)))

    # Filter to get only unique folds (avoid duplicates)
    checkpoint_files = checkpoint_files[:n_folds] if len(checkpoint_files) >= n_folds else checkpoint_files

    if not checkpoint_files:
        print(f"\n❌ Error: No fold checkpoints found for {model_name}")
        print(f"   Looking for pattern: {checkpoint_pattern}")
        print(f"   Please run main_full.py first to train the model.")
        return

    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for i, ckpt in enumerate(checkpoint_files, 1):
        print(f"  {i}. {ckpt.name}")

    # ============================================
    # 2. Load models
    # ============================================
    from src.model import get_model

    config = DocumentConfig()
    config.BATCH_SIZE = batch_size
    device = config.DEVICE

    train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_data(config)

    if test_dataset is None:
        print("\n❌ Error: No test data found!")
        return

    print(f"\nLoading {len(checkpoint_files)} models...")

    # Create fold_results structure
    fold_results = []

    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"\n  Loading fold {i+1}/{len(checkpoint_files)}...")

        # Create model
        model = get_model(model_name, num_classes=num_classes, pretrained=False, dropout_rate=0.0)

        # Load checkpoint
        try:
            state_dict = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()

            # Extract F1 score from filename
            import re
            match = re.search(r'f1_([\d.]+)', checkpoint_file.stem, re.IGNORECASE)
            val_f1 = float(match.group(1)) if match else 0.0

            fold_results.append({
                'fold': i + 1,
                'best_model_state': state_dict,
                'best_val_f1': val_f1,
                'best_epoch': 0  # Unknown
            })

            print(f"    ✅ Loaded: {checkpoint_file.name} (Val F1: {val_f1:.4f})")

        except Exception as e:
            print(f"    ❌ Error loading {checkpoint_file.name}: {e}")
            continue

    if not fold_results:
        print("\n❌ Error: No models loaded successfully!")
        return

    print(f"\n✅ Successfully loaded {len(fold_results)} models")

    # ============================================
    # 3. Ensemble Prediction
    # ============================================
    print("\n" + "="*70)
    print("Running K-Fold Ensemble Prediction")
    print("="*70)

    tta_transforms = ['original', 'hflip', 'vflip', 'rotate90'] if use_tta else ['original']

    _, test_f1, predictions, _ = evaluate_ensemble(
        fold_results=fold_results,
        test_dataset=test_dataset,
        config=config,
        use_tta=use_tta,
        tta_transforms=tta_transforms
    )

    # ============================================
    # 4. Save Submission
    # ============================================
    print("\n" + "="*70)
    print("Saving Submission File")
    print("="*70)

    submission_dir = Path('submissions')
    submission_dir.mkdir(exist_ok=True)

    suffix = f'kfold_{model_name}_tta' if use_tta else f'kfold_{model_name}'

    save_submission(
        preds=predictions,
        sample_path=config.SUBMISSION_PATH,
        save_path=submission_dir,
        f1_score=test_f1,
        suffix=suffix
    )

    # ============================================
    # 5. Summary
    # ============================================
    print("\n" + "="*70)
    print("K-Fold Ensemble Complete!")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Number of folds: {len(fold_results)}")
    print(f"  TTA: {'Yes' if use_tta else 'No'}")
    print(f"  Average CV F1: {test_f1:.4f}")

    print("="*70)


if __name__ == "__main__":
    try:
        fire.Fire({
            'from_pkl': submit_from_fold_results,
            'from_checkpoints': submit_from_checkpoints,
        })
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
