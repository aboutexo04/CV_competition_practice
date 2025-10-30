from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Configs and model display names are imported inside the function to avoid
# import-time side effects when this module is imported early by notebooks.


def _ensure_header(file_path: Path) -> None:
    if file_path.exists():
        return
    header = (
        "# 실험 결과 모음\n\n"
        "이 파일은 모든 실험 결과를 자동으로 기록합니다.\n\n"
        "## 실험 목록\n\n"
    )
    file_path.write_text(header, encoding="utf-8")


def log_experiment_results(
    project_root: Path | str,
    fold_results: List[Dict[str, Any]],
    results: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None,
) -> Path:
    """
    Append a formatted experiment result section to experiment_results.md.

    Args:
        project_root: Project root directory (Path or str).
        fold_results: List of fold dicts containing at least 'fold' and 'best_val_f1'.
        results: Optional dict containing 'test_acc' and 'test_f1'.
        experiment_name: Optional explicit experiment name. If None, derive from config or wandb.

    Returns:
        Path to the written markdown file.
    """
    from src.config import (
        MODEL_NAME,
        BATCH_SIZE,
        EPOCHS,
        LR,
        N_FOLDS,
        IMAGE_SIZE,
        USE_SUBSET,
        SUBSET_RATIO,
    )
    from src.model import MODEL_DISPLAY_NAME, SELECTED_MODEL

    prj_root = Path(project_root)
    results_md_path = prj_root / "experiment_results.md"
    _ensure_header(results_md_path)

    # Derive experiment name
    if experiment_name is None:
        try:
            import wandb  # type: ignore

            experiment_name = (
                wandb.run.name if getattr(wandb, "run", None) is not None else f"{MODEL_NAME}_bs{BATCH_SIZE}_ep{EPOCHS}"
            )
        except Exception:
            experiment_name = f"{MODEL_NAME}_bs{BATCH_SIZE}_ep{EPOCHS}"

    # CV stats
    cv_vals = [
        float(fr.get("best_val_f1"))
        for fr in (fold_results or [])
        if isinstance(fr, dict) and fr.get("best_val_f1") is not None
    ]
    avg_f1 = float(np.mean(cv_vals)) if cv_vals else None
    std_f1 = float(np.std(cv_vals)) if cv_vals else None

    # Test stats
    test_acc = None
    test_f1 = None
    if isinstance(results, dict):
        if results.get("test_acc") is not None:
            test_acc = float(results["test_acc"])  # percentage
        if results.get("test_f1") is not None:
            test_f1 = float(results["test_f1"])  # macro f1

    data_mode = f"{int(SUBSET_RATIO * 100)}% 데이터 (연습용)" if USE_SUBSET else "전체 데이터"

    lines: List[str] = []
    lines.append(f"### {experiment_name}\n\n")
    lines.append(f"**실행 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    lines.append(f"**데이터 모드**: {data_mode}  \n")
    lines.append(f"**모델**: {MODEL_DISPLAY_NAME} ({SELECTED_MODEL})  \n\n")

    lines.append("**하이퍼파라미터**:  \n")
    lines.append(f"- Image Size: {IMAGE_SIZE}  \n")
    lines.append(f"- Epochs: {EPOCHS}  \n")
    lines.append(f"- Batch Size: {BATCH_SIZE}  \n")
    lines.append(f"- Learning Rate: {LR}  \n")
    lines.append(f"- K-Fold: {N_FOLDS} folds  \n\n")

    if cv_vals:
        lines.append("**Validation Results**:  \n")
        for fr in fold_results:
            if isinstance(fr, dict) and fr.get("fold") is not None and fr.get("best_val_f1") is not None:
                lines.append(f"- Fold {int(fr['fold'])}: Val F1 = {float(fr['best_val_f1']):.4f}  \n")
        if avg_f1 is not None and std_f1 is not None:
            lines.append(f"- Average: {avg_f1:.4f} ± {std_f1:.4f}  \n\n")

    if (test_acc is not None) or (test_f1 is not None):
        lines.append("**Test Results**:  \n")
        if test_acc is not None:
            lines.append(f"- Accuracy: {test_acc:.2f}%  \n")
        if test_f1 is not None:
            lines.append(f"- Macro F1: {test_f1:.4f}  \n\n")

    lines.append("-----------------------------------------------------\n\n")

    with results_md_path.open("a", encoding="utf-8") as f:
        f.writelines(lines)

    # 출력: 한 문장만
    print("실험 결과가 experiment_results.md 파일에 저장되었습니다")
