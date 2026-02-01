"""
Merged inference:
1) Predict algorithm/family label from a QASM file (using predict_algorithm.py logic)
2) Feed the predicted family into inference.py to get threshold + runtime predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

import inference
import predict_algorithm


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_idx_to_label(exp_dir: Path) -> dict[str, str]:
    important = exp_dir / "model_important_stuff.json"
    if not important.exists():
        raise FileNotFoundError(f"Missing idx-to-label file: {important}")
    data = json.loads(important.read_text(encoding="utf-8"))
    idx_to_label = data.get("label_map", {}).get("idx_to_label")
    if not isinstance(idx_to_label, dict) or not idx_to_label:
        raise ValueError(f"Could not read idx_to_label from {important}")
    return {str(k): str(v) for k, v in idx_to_label.items()}


def _label_to_family(label: str) -> str:
    norm = label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "dj": "Deutsch_Jozsa",
        "ghz": "GHZ",
        "graphstate": "GraphState",
        "qaoa": "QAOA",
        "qft": "QFT",
        "qftentangled": "QFT_Entangled",
        "qnn": "QNN",
        "qpeexact": "QPE_Exact",
        "wstate": "W_State",
        "other": "unknown",
    }
    return aliases.get(norm, "unknown")


def predict_family_from_qasm(
    *,
    exp_dir: Path,
    qasm: str,
    reject_threshold: float | None,
    reject_label: str,
) -> str:
    config_path = Path("config_algorithm.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing algorithm config: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    model_path = exp_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing algorithm checkpoint: {model_path}")

    feature_stats_path = exp_dir / "feature_stats.json"
    if not feature_stats_path.exists():
        feature_stats_path = None  # type: ignore[assignment]

    idx_to_label = _load_idx_to_label(exp_dir)

    device = _auto_device()
    model, registry, _ = predict_algorithm.load_model(
        str(model_path),
        config,
        label_map_path=None,
        device=device,
        feature_stats_path=(str(feature_stats_path) if feature_stats_path is not None else None),
    )

    use_graph = bool(config.get("use_graph_features", True))
    best_label, confidence = predict_algorithm.predict_single_qasm(
        model,
        registry,
        idx_to_label,
        qasm,
        device,
        use_graph,
    )

    label, _rejected = predict_algorithm.apply_rejection(
        best_label,
        confidence,
        reject_threshold,
        reject_label,
    )
    return _label_to_family(label)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Algorithm->family->threshold/runtime inference")
    parser.add_argument("--exp-dir", type=str, required=True, help="Algorithm classifier exp dir (has best_model.pt)")
    parser.add_argument("--qasm", type=str, required=True, help="Path to a single QASM file")
    parser.add_argument("--config", type=str, required=True, help="Threshold/runtime model config YAML (no defaults)")
    parser.add_argument("--model-path", type=str, required=True, help="Threshold/runtime model checkpoint (.pt)")
    parser.add_argument(
        "--runtime-threshold",
        type=int,
        required=True,
        choices=inference.THRESHOLD_LADDER,
        help="Threshold value to feed to runtime head (no defaults)",
    )
    parser.add_argument(
        "--algo-reject-threshold",
        type=float,
        default=None,
        help="If set, treat algorithm prediction as rejected when confidence < threshold.",
    )
    parser.add_argument(
        "--algo-reject-label",
        type=str,
        default="other",
        help="Label to use when rejecting (defaults to 'other' -> family 'unknown').",
    )
    args = parser.parse_args(argv)

    family = predict_family_from_qasm(
        exp_dir=Path(args.exp_dir),
        qasm=args.qasm,
        reject_threshold=args.algo_reject_threshold,
        reject_label=args.algo_reject_label,
    )
    disable_family = family == "unknown"

    preds = inference.predict_threshold_and_runtime(
        args.qasm,
        config_path=args.config,
        model_path=args.model_path,
        family=family,
        runtime_threshold=args.runtime_threshold,
        disable_family=disable_family,
    )

    for p in preds:
        print(
            f"{p.file} ({p.backend}, {p.precision}) -> "
            f"threshold={p.predicted_threshold_min}, runtime_s={p.predicted_forward_wall_s:.6f}"
        )


if __name__ == "__main__":
    main()
