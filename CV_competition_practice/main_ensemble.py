#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Script for Final Submission

Combines predictions from multiple trained models for better performance.

Usage:
    # Ensemble with default models
    python main_ensemble.py

    # Ensemble with Fire CLI
    python main_ensemble.py --models efficientnet_b0,efficientnet_b1,resnet50
    python main_ensemble.py --models efficientnet_b0,resnet50 --weights 0.6,0.4
    python main_ensemble.py --ensemble_method=voting
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
from src.data import load_data, get_val_augmentation
from src.model import get_model
from src.utils import set_seed
from torch.utils.data import DataLoader


def load_trained_models(model_names, num_classes, device, models_dir='models'):
    """
    Load trained models from checkpoint files

    Args:
        model_names: List of model names to load
        num_classes: Number of classes
        device: Device to load models on
        models_dir: Directory containing saved model checkpoints

    Returns:
        List of loaded models
    """
    models = []
    models_path = Path(models_dir)

    print(f"\nLoading {len(model_names)} models...")

    for model_name in model_names:
        print(f"\n  Loading {model_name}...")

        # Create model
        model = get_model(model_name, num_classes=num_classes, pretrained=False)

        # Find checkpoint file
        checkpoint_pattern = f"*{model_name}*.pth"
        checkpoint_files = list(models_path.glob(checkpoint_pattern))

        if not checkpoint_files:
            print(f"    Warning: No checkpoint found for {model_name}")
            print(f"    Creating model with random weights (for testing)")
            model = model.to(device)
            model.eval()
            models.append(model)
            continue

        # Load best checkpoint (if multiple exist)
        checkpoint_file = checkpoint_files[0]
        if len(checkpoint_files) > 1:
            print(f"    Found {len(checkpoint_files)} checkpoints, using {checkpoint_file.name}")

        # Load state dict
        try:
            state_dict = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(state_dict)
            print(f"    Loaded checkpoint: {checkpoint_file.name}")
        except Exception as e:
            print(f"    Error loading checkpoint: {e}")
            print(f"    Using model with random weights")

        model = model.to(device)
        model.eval()
        models.append(model)

    print(f"\nSuccessfully loaded {len(models)} models")
    return models


def ensemble_predict(models, dataloader, device, method='average', weights=None):
    """
    Make ensemble predictions

    Args:
        models: List of models
        dataloader: DataLoader for test data
        device: Device
        method: Ensemble method ('average' or 'voting')
        weights: Weights for weighted average (None for equal weights)

    Returns:
        predictions: Ensemble predictions (numpy array)
        all_probs: Individual model probabilities (for analysis)
    """
    num_models = len(models)

    # Set equal weights if not provided
    if weights is None:
        weights = [1.0 / num_models] * num_models
    else:
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

    print(f"\nEnsemble method: {method}")
    print(f"Model weights: {weights}")

    all_predictions = []
    all_model_probs = [[] for _ in range(num_models)]

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Ensemble Prediction"):
            images = images.to(device)

            # Get predictions from all models
            batch_probs = []
            for i, model in enumerate(models):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())

                # Store for analysis
                all_model_probs[i].append(probs.cpu().numpy())

            # Ensemble predictions
            if method == 'average':
                # Weighted average of probabilities
                ensemble_probs = np.zeros_like(batch_probs[0])
                for prob, weight in zip(batch_probs, weights):
                    ensemble_probs += prob * weight
                predictions = np.argmax(ensemble_probs, axis=1)

            elif method == 'voting':
                # Majority voting
                batch_preds = [np.argmax(prob, axis=1) for prob in batch_probs]
                batch_preds = np.array(batch_preds)  # (num_models, batch_size)
                predictions = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(),
                    axis=0,
                    arr=batch_preds
                )

            else:
                raise ValueError(f"Unknown ensemble method: {method}")

            all_predictions.extend(predictions)

    # Concatenate all model probabilities
    all_model_probs = [np.concatenate(probs, axis=0) for probs in all_model_probs]

    return np.array(all_predictions), all_model_probs


def run_ensemble(
    models=None,
    weights=None,
    ensemble_method='average',
    batch_size=32,
    image_size=224,
    seed=42,
    output_file='submission_ensemble.csv',
):
    """
    Run ensemble prediction on test data

    Args:
        models: Comma-separated model names (e.g., 'efficientnet_b0,resnet50')
        weights: Comma-separated weights for models (e.g., '0.6,0.4')
        ensemble_method: Ensemble method ('average' or 'voting')
        batch_size: Batch size for inference
        image_size: Image size
        seed: Random seed
        output_file: Output submission file name
    """

    # ============================================
    # 1. Setup
    # ============================================
    print("\n" + "="*70)
    print("Ensemble Prediction Start")
    print("="*70)

    # Parse model names
    if models is None:
        # Default models
        model_names = ['efficientnet_b0', 'efficientnet_b1', 'resnet50']
        print("\nUsing default models:")
    else:
        model_names = [m.strip() for m in models.split(',')]
        print(f"\nUsing specified models:")

    for i, name in enumerate(model_names, 1):
        print(f"  {i}. {name}")

    # Parse weights
    if weights is not None:
        weights = [float(w.strip()) for w in weights.split(',')]
        if len(weights) != len(model_names):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(model_names)})")
    else:
        weights = None
        print(f"\nUsing equal weights for all models")

    # Set seed
    set_seed(seed)

    # ============================================
    # 2. Load Config and Data
    # ============================================
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)

    config = DocumentConfig()
    config.BATCH_SIZE = batch_size
    config.IMAGE_SIZE = image_size

    train_dataset_raw, test_dataset, train_labels, class_names, num_classes = load_data(config)

    if test_dataset is None:
        print("\nError: No test data found!")
        print("Please ensure test data is available in data/document/ directory")
        return

    print(f"\nTest samples: {len(test_dataset):,}")
    print(f"Classes ({num_classes}): {class_names}")

    device = config.DEVICE
    print(f"Device: {device}")

    # ============================================
    # 3. Load Trained Models
    # ============================================
    print("\n" + "="*70)
    print("Loading Trained Models")
    print("="*70)

    models_list = load_trained_models(
        model_names=model_names,
        num_classes=num_classes,
        device=device
    )

    # ============================================
    # 4. Create DataLoader
    # ============================================
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ============================================
    # 5. Ensemble Prediction
    # ============================================
    print("\n" + "="*70)
    print("Running Ensemble Prediction")
    print("="*70)

    predictions, all_model_probs = ensemble_predict(
        models=models_list,
        dataloader=test_loader,
        device=device,
        method=ensemble_method,
        weights=weights
    )

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")

    # ============================================
    # 6. Analyze Ensemble Agreement
    # ============================================
    print("\n" + "="*70)
    print("Ensemble Analysis")
    print("="*70)

    # Calculate agreement between models
    individual_preds = [np.argmax(probs, axis=1) for probs in all_model_probs]
    agreement_count = np.zeros(len(predictions))

    for i in range(len(predictions)):
        votes = [preds[i] for preds in individual_preds]
        agreement_count[i] = votes.count(predictions[i])

    print(f"\nModel Agreement Statistics:")
    print(f"  All models agree: {np.sum(agreement_count == len(models_list)):,} samples ({np.mean(agreement_count == len(models_list))*100:.1f}%)")
    print(f"  Majority agree: {np.sum(agreement_count >= len(models_list)/2):,} samples ({np.mean(agreement_count >= len(models_list)/2)*100:.1f}%)")
    print(f"  Average agreement: {np.mean(agreement_count):.2f} / {len(models_list)} models")

    # ============================================
    # 7. Save Submission
    # ============================================
    print("\n" + "="*70)
    print("Saving Submission File")
    print("="*70)

    from src.submission import save_submission

    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_file

    save_submission(
        preds=predictions,
        sample_path='sample_submission.csv',
        save_path=str(output_path)
    )

    # ============================================
    # 8. Summary
    # ============================================
    print("\n" + "="*70)
    print("Ensemble Complete!")
    print("="*70)

    print(f"\nEnsemble Configuration:")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Method: {ensemble_method}")
    if weights:
        print(f"  Weights: {weights}")
    print(f"  Total predictions: {len(predictions):,}")
    print(f"  Output file: {output_path}")

    print("\nClass Distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(predictions == i)
        percentage = count / len(predictions) * 100
        print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

    print("="*70)


if __name__ == "__main__":
    try:
        fire.Fire(run_ensemble)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
