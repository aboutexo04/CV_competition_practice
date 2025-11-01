#!/usr/bin/env python3
"""
Full Training Script with Fire CLI

Usage:
    # Default settings (DocumentConfig)
    python main_full.py

    # Change specific parameters
    python main_full.py --epochs=50 --batch_size=64
    python main_full.py --model_name=resnet50 --lr=0.0001
    python main_full.py --n_folds=10 --use_subset=False

    # Multiple parameters
    python main_full.py --model_name=efficientnet_b3 --epochs=100 --batch_size=32 --lr=0.0001

    # Enable Wandb
    python main_full.py --use_wandb=True --wandb_project=my-project

    # Quick test mode
    python main_full.py --use_subset=True --subset_ratio=0.1 --epochs=5 --n_folds=2

    # Create submission with TTA
    python main_full.py --use_tta=True --create_submission=True

    # Disable submission creation
    python main_full.py --create_submission=False
"""

import sys
from pathlib import Path
import torch
import warnings
import fire
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
from src.train import run_kfold_training, train_on_full_data
from src.evaluation import run_full_evaluation, evaluate_ensemble
from src.logger import log_experiment_results
from src.submission import save_submission
from src.utils import set_seed
import pickle


def train(
    # Model settings
    model_name='efficientnet_b0',

    # Data settings
    dataset_type='document',
    image_size=224,
    batch_size=32,
    use_subset=False,
    subset_ratio=1.0,

    # Augmentation settings
    aug_strategy='hybrid',
    augraphy_strength='light',

    # Training settings
    epochs=20,
    lr=0.0001,
    patience=5,
    early_stopping_delta=0,
    n_folds=5,

    # Advanced settings
    use_label_smoothing=False,
    label_smoothing_factor=0.1,
    use_tta=False,
    use_balanced_sampling=True,
    train_final_model=True,
    dropout_rate=0.3,
    weight_decay=5e-4,

    # Ensemble settings
    use_best_k_folds=False,
    best_k_count=4,

    # Submission settings
    create_submission=True,
    save_fold_results=True,

    # Wandb settings
    use_wandb=False,
    wandb_project='document-classification',

    # Other settings
    seed=42,
    num_workers=0,
):
    """
    Train model with specified parameters

    Args:
        model_name: Model architecture name
        dataset_type: Dataset type ('document')
        image_size: Image size for training
        batch_size: Batch size for training
        use_subset: Use subset of data for quick testing
        subset_ratio: Ratio of data to use if use_subset=True
        aug_strategy: Augmentation strategy ('auto', 'albumentations', 'augraphy', 'hybrid')
        augraphy_strength: Augraphy augmentation strength ('light', 'medium', 'heavy')
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        early_stopping_delta: Minimum improvement to count as improvement (default: 0.001)
        n_folds: Number of K-Fold splits
        use_label_smoothing: Use label smoothing loss
        label_smoothing_factor: Label smoothing factor (0.0-1.0)
        use_tta: Use Test Time Augmentation
        use_balanced_sampling: Use class-balanced sampling (WeightedRandomSampler)
        train_final_model: Create final model submission (model is always trained and saved)
        dropout_rate: Dropout rate for model (0.0-0.5 recommended)
        weight_decay: L2 regularization weight decay (1e-5 to 1e-3 recommended)
        use_best_k_folds: Use only top K folds by validation F1 for ensemble (default: False)
        best_k_count: Number of best folds to use when use_best_k_folds=True (default: 4)
        create_submission: Create submission file after training
        save_fold_results: Save fold results to file
        use_wandb: Enable Wandb logging
        wandb_project: Wandb project name
        seed: Random seed
        num_workers: DataLoader num_workers
    """

    # ============================================
    # 1. Config Setup
    # ============================================
    print("\n" + "="*70)
    print("Full Training Start")
    print("="*70)

    config = DocumentConfig()

    # Update config with CLI arguments
    config.update(
        MODEL_NAME=model_name,
        DATASET_TYPE=dataset_type,
        IMAGE_SIZE=image_size,
        BATCH_SIZE=batch_size,
        USE_SUBSET=use_subset,
        SUBSET_RATIO=subset_ratio,
        AUG_STRATEGY=aug_strategy,
        AUGRAPHY_STRENGTH=augraphy_strength,
        EPOCHS=epochs,
        LR=lr,
        PATIENCE=patience,
        EARLY_STOPPING_DELTA=early_stopping_delta,
        N_FOLDS=n_folds,
        USE_LABEL_SMOOTHING=use_label_smoothing,
        LABEL_SMOOTHING_FACTOR=label_smoothing_factor,
        USE_TTA=use_tta,
        USE_CLASS_BALANCED_SAMPLING=use_balanced_sampling,
        TRAIN_FINAL_MODEL=train_final_model,
        DROPOUT_RATE=dropout_rate,
        WEIGHT_DECAY=weight_decay,
        USE_WANDB=use_wandb,
        WANDB_PROJECT=wandb_project,
        SEED=seed,
        NUM_WORKERS=num_workers,
    )

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
    # 3. Initialize Wandb (if enabled)
    # ============================================
    if config.USE_WANDB:
        import wandb
        from src.model import MODEL_CONFIGS

        # Model display name
        model_display = MODEL_CONFIGS.get(
            config.MODEL_NAME,
            {'display_name': config.MODEL_NAME}
        )['display_name']

        # Run name
        run_name = f"{config.MODEL_NAME}_bs{config.BATCH_SIZE}_ep{config.EPOCHS}"
        if config.USE_SUBSET:
            run_name += f"_sub{int(config.SUBSET_RATIO*100)}"

        # Initialize Wandb
        wandb.init(
            project=config.WANDB_PROJECT,
            name=run_name,
            config=config.to_dict()
        )

        print(f"\nWandb initialized: {wandb.run.name}")
        print(f"Project: {config.WANDB_PROJECT}")
    else:
        print("\nWandb disabled")

    # ============================================
    # 4. K-Fold Training
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
    # 5. Train Final Model on Full Data (always)
    # ============================================
    final_model_result = train_on_full_data(
        train_dataset_raw=train_dataset_raw,
        train_labels=train_labels,
        fold_results=fold_results,
        config=config
    )

    # ============================================
    # 6. Evaluation (if test data exists)
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
    # 7. Log Experiment Results
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
    # 8. Wandb Finish
    # ============================================
    if config.USE_WANDB:
        import wandb
        wandb.finish()
        print("\nWandb finished")

    # ============================================
    # 9. Save Fold Results (optional)
    # ============================================
    if save_fold_results:
        print("\n" + "="*70)
        print("Saving Fold Results")
        print("="*70)

        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)

        fold_results_path = results_dir / 'fold_results.pkl'
        with open(fold_results_path, 'wb') as f:
            pickle.dump(fold_results, f)

        print(f"✅ Fold results saved: {fold_results_path}")

    # ============================================
    # 10. Create Submission File (optional)
    # ============================================
    if create_submission and test_dataset is not None:
        print("\n" + "="*70)
        print("Creating Submission Files")
        print("="*70)

        submission_dir = Path('submissions')
        submission_dir.mkdir(exist_ok=True)

        # ============================================
        # 10-1. K-Fold Ensemble Submission
        # ============================================
        print("\n" + "="*50)
        print("📊 K-Fold Ensemble Submission")
        print("="*50)

        # Create submissions (standard and optionally TTA)
        tta_list = [False, True] if use_tta else [False]

        for use_tta_flag in tta_list:
            tta_label = "with TTA" if use_tta_flag else "standard"
            print(f"\n{'🔄' if use_tta_flag else '📝'} Creating K-Fold ensemble {tta_label} submission...")

            _, test_f1, predictions, _ = evaluate_ensemble(
                fold_results=fold_results,
                test_dataset=test_dataset,
                config=config,
                use_tta=use_tta_flag,
                tta_transforms=['original', 'hflip', 'vflip', 'rotate90'] if use_tta_flag else ['original'],
                top_k_folds=best_k_count if use_best_k_folds else None
            )

            try:
                save_submission(
                    preds=predictions,
                    sample_path=config.SUBMISSION_PATH,
                    save_path=submission_dir,
                    f1_score=test_f1,
                    suffix='kfold'
                )
            except Exception as e:
                print(f"❌ Error: {e}")
                if not use_tta_flag:  # Stop if standard submission fails
                    return

        # ============================================
        # 10-2. Final Model (Full Data) Submission
        # ============================================
        if config.TRAIN_FINAL_MODEL and final_model_result is not None:
            print("\n" + "="*50)
            print("🚀 Final Model (100% data) Submission")
            print("="*50)

            from src.evaluation import evaluate_single_model

            for use_tta_flag in tta_list:
                tta_label = "with TTA" if use_tta_flag else "standard"
                print(f"\n{'🔄' if use_tta_flag else '📝'} Creating final model {tta_label} submission...")

                _, test_f1, predictions, _ = evaluate_single_model(
                    model=final_model_result['model'],
                    test_dataset=test_dataset,
                    config=config,
                    use_tta=use_tta_flag,
                    tta_transforms=['original', 'hflip', 'vflip', 'rotate90'] if use_tta_flag else ['original']
                )

                try:
                    save_submission(
                        preds=predictions,
                        sample_path=config.SUBMISSION_PATH,
                        save_path=submission_dir,
                        f1_score=final_model_result['avg_val_f1'],  # CV F1 참고용
                        suffix='final'
                    )
                except Exception as e:
                    print(f"❌ Error: {e}")

    # ============================================
    # 11. Summary
    # ============================================
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

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

    print("="*70)


if __name__ == "__main__":
    try:
        fire.Fire(train)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
