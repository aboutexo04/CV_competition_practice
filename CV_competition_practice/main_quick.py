#!/usr/bin/env python3
"""
Quick Test Script for Fast Experiments

Settings:
- K-Fold: 2
- Epochs: 2
- Subset: 10%
- Purpose: Code validation and quick experiments
"""

import sys
from pathlib import Path
import torch
import warnings
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
from src.config import QuickTestConfig
from src.data import load_data
from src.train import run_kfold_training
from src.evaluation import run_full_evaluation
from src.logger import log_experiment_results
from src.utils import set_seed

def main():
    """Main execution function"""

    # ============================================
    # 1. Config Setup
    # ============================================
    print("\n" + "="*70)
    print("Quick Test Start")
    print("="*70)

    config = QuickTestConfig()

    # Ensure quick test settings
    config.EPOCHS = 2
    config.N_FOLDS = 2
    config.USE_SUBSET = False
    config.USE_WANDB = False
    config.USE_TTA = False

    # time stamp
    import time
    start_time = time.time()

    # Print config
    config.print_config()

    # Set seed
    set_seed(config.SEED)

    # ============================================
    # 2. Load Data
    # ============================================
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)

    train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_data(config)

    # Update config
    config.NUM_CLASSES = num_classes

    print(f"\nData loaded successfully")
    print(f"Device: {config.DEVICE}")
    print(f"Train samples: {len(train_dataset_raw):,}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes ({num_classes}): {class_names}")

    # ============================================
    # 3. K-Fold Training
    # ============================================
    print("\n" + "="*70)
    print("K-Fold Training Start")
    print("="*70)

    fold_results = run_kfold_training(
        train_dataset_raw=train_dataset_raw,
        train_labels=train_labels,
        config=config
    )

    print("\n" + "="*70)
    print("K-Fold Training Complete!")
    print("="*70)

    # ============================================
    # 4. Evaluation (if test data exists)
    # ============================================
    if test_dataset is not None:
        print("\n" + "="*70)
        print("Test Data Evaluation")
        print("="*70)

        results = run_full_evaluation(
            fold_results=fold_results,
            test_dataset=test_dataset,
            class_names=class_names,
            config=config
        )

        print("\n" + "="*70)
        print("Final Evaluation Results")
        print("="*70)
        print(f"Final Test Accuracy: {results['test_acc']:.2f}%")
        print(f"Final Test F1 Score: {results['test_f1']:.4f}")
        print("="*70)
    else:
        print("\nNo test data - skipping evaluation")
        results = None

    # ============================================
    # 5. Log Experiment Results
    # ============================================
    print("\n" + "="*70)
    print("Saving Experiment Results")
    print("="*70)

    log_experiment_results(
        fold_results=fold_results,
        results=results,
        config=config
    )

    # ============================================
    # 6. Summary
    # ============================================
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("Quick Test Complete!")
    print("="*70)

    #시간출력
    elapsed_time = time.time() - start_time    
    print(f"Elapsed Time: {elapsed_time/60:.1f} minutes")

    # Fold results summary
    print("\nFold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Val F1 = {result['best_val_f1']:.4f}")

    import numpy as np
    avg_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in fold_results])

    print(f"\nAverage Validation F1: {avg_f1:.4f} +/- {std_f1:.4f}")

    if results:
        print(f"\nTest Performance:")
        print(f"  Accuracy: {results['test_acc']:.2f}%")
        print(f"  F1 Score: {results['test_f1']:.4f}")

    print("\nNext Steps:")
    print("  1. Verify settings are working correctly")
    print("  2. Validate augmentation effects")
    print("  3. Run full training with main_full.py")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
