"""
Prediction script for algorithm/family classification.

Usage (single file):
    python predict_algorithm.py \
        --qasm circuits/mqtbench_large/val/qaoa_indep_qiskit_15.qasm \
        --model-path exp/default/best_model.pt \
        --config config_algorithm.yaml \
        --label-map data/mqtbench_large_label_map.json

Usage (directory):
    python predict_algorithm.py \
        --qasm-dir circuits/mqtbench_large/val \
        --out predictions_algorithm.json \
        --model-path exp/default/best_model.pt \
        --config config_algorithm.yaml \
        --label-map data/mqtbench_large_label_map.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.dataset_algorithm import load_label_map
from src.feature_extractors import (
    create_default_registry,
    create_default_no_context_registry,
    create_extended_registry,
    create_minimal_registry,
)
from src.model_algorithm import create_algorithm_model
from src.features import qasm_file_to_features


def load_feature_stats(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_feature_registry(config):
    registry_name = config.get("feature_registry", "default_no_context")
    if registry_name == "default":
        return create_default_registry()
    if registry_name == "default_no_context":
        return create_default_no_context_registry()
    if registry_name == "minimal":
        return create_minimal_registry()
    if registry_name == "extended":
        return create_extended_registry()
    raise ValueError(f"Unknown feature_registry: {registry_name}")


def load_model(model_path, config, label_map_path, device, feature_stats_path=None):
    idx_to_label = {}
    if label_map_path:
        label_map, _, label_to_idx, idx_to_label = load_label_map(label_map_path)
        num_classes = int(label_map.get("num_classes", len(label_to_idx)))
        config["num_classes"] = num_classes

    registry = create_feature_registry(config)
    config["global_feature_dim"] = registry.total_dim()

    if feature_stats_path is None:
        exp_dir = config.get("exp_dir")
        if exp_dir:
            candidate = Path(exp_dir) / "feature_stats.json"
            if candidate.exists():
                feature_stats_path = candidate

    if feature_stats_path is not None:
        stats = load_feature_stats(Path(feature_stats_path))
        if "mean" in stats and "std" in stats:
            registry.set_standardization(stats["mean"], stats["std"])

    checkpoint = torch.load(model_path, map_location=device)
    if not label_map_path:
        ckpt_config = checkpoint.get("config", {})
        if "num_classes" in ckpt_config:
            config["num_classes"] = int(ckpt_config["num_classes"])
        else:
            # Fall back to classifier head shape
            state = checkpoint.get("model_state_dict", {})
            head_weight = state.get("class_head.weight")
            if head_weight is not None:
                config["num_classes"] = int(head_weight.shape[0])
    model = create_algorithm_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, registry, idx_to_label


def predict_single_qasm(model, registry, idx_to_label, qasm_path, device, use_graph):
    features = qasm_file_to_features(
        qasm_path,
        backend="CPU",
        precision="double",
        registry=registry,
        augment=False,
    )

    from torch_geometric.data import Data

    if use_graph and "gate_type_idx" in features:
        data = Data(
            gate_type_idx=features["gate_type_idx"],
            qubit_indices=features["qubit_indices"],
            is_directional=features["is_directional"],
            gate_arity=features["gate_arity"],
            gate_index_norm=features["gate_index_norm"],
            edge_index=features["edge_index"],
            global_features=features["global_features"],
        )
        data.batch = torch.zeros(features["gate_type_idx"].size(0), dtype=torch.long)
    else:
        data = Data(global_features=features["global_features"])

    data = data.to(device)

    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs.max(dim=1).values.item())

    label = idx_to_label.get(str(pred_idx), idx_to_label.get(pred_idx, str(pred_idx)))
    return label, confidence


def apply_rejection(predicted_label: str, confidence: float, reject_threshold: float | None, reject_label: str):
    if reject_threshold is None:
        return predicted_label, False
    if confidence < reject_threshold:
        return reject_label, True
    return predicted_label, False


def main():
    parser = argparse.ArgumentParser(description="Predict algorithm labels for QASM files")
    parser.add_argument("--qasm", type=str, default=None, help="Path to a single QASM file")
    parser.add_argument(
        "--qasm-dir",
        type=str,
        default=None,
        help="Directory of QASM files (predict all *.qasm)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path for batch predictions",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained algorithm classifier checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_algorithm.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Path to label map JSON (optional; used only for idx->label names)",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Experiment directory (auto-load best_model.pt and feature stats)",
    )
    parser.add_argument(
        "--idx-to-label",
        type=str,
        default=None,
        help="Path to JSON with idx_to_label mapping (optional)",
    )
    parser.add_argument(
        "--feature-stats",
        type=str,
        default=None,
        help="Path to feature_stats.json (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detect if not specified)",
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=None,
        help="If set, output --reject-label when max-softmax confidence is below this threshold",
    )
    parser.add_argument(
        "--reject-label",
        type=str,
        default="other",
        help="Label to output when rejecting (used only with --reject-threshold)",
    )
    args = parser.parse_args()

    if not args.qasm and not args.qasm_dir:
        raise SystemExit("Provide --qasm or --qasm-dir")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        if args.model_path is None:
            args.model_path = str(exp_dir / "best_model.pt")
        if args.feature_stats is None:
            candidate = exp_dir / "feature_stats.json"
            if candidate.exists():
                args.feature_stats = str(candidate)
        if args.idx_to_label is None:
            candidate = exp_dir / "model_important_stuff.json"
            if candidate.exists():
                args.idx_to_label = str(candidate)

    if args.model_path is None:
        raise SystemExit("Provide --model-path or --exp-dir")

    idx_to_label = {}
    label_map_path = args.label_map
    if args.idx_to_label:
        with open(args.idx_to_label, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx_to_label = data.get("label_map", {}).get("idx_to_label", data.get("idx_to_label", data))
        idx_to_label = {str(k): v for k, v in idx_to_label.items()}
        label_map_path = None

    model, registry, loaded_idx_to_label = load_model(
        args.model_path,
        config,
        label_map_path,
        device,
        feature_stats_path=args.feature_stats,
    )
    if loaded_idx_to_label:
        idx_to_label = loaded_idx_to_label

    use_graph = config.get("use_graph_features", True)

    if args.qasm:
        best_label, confidence = predict_single_qasm(model, registry, idx_to_label, args.qasm, device, use_graph)
        label, rejected = apply_rejection(best_label, confidence, args.reject_threshold, args.reject_label)
        extra = f", best={best_label}" if rejected else ""
        print(f"{args.qasm} -> {label} (confidence={confidence:.6f}{extra})")
        return

    qasm_dir = Path(args.qasm_dir)
    if not qasm_dir.exists():
        raise SystemExit(f"Directory not found: {qasm_dir}")

    predictions = []
    for qasm_path in sorted(qasm_dir.glob("*.qasm")):
        best_label, confidence = predict_single_qasm(model, registry, idx_to_label, qasm_path, device, use_graph)
        label, rejected = apply_rejection(best_label, confidence, args.reject_threshold, args.reject_label)
        predictions.append(
            {
                "qasm": qasm_path.name,
                "predicted_label": label,
                "confidence": confidence,
                "rejected": rejected,
                "best_label": best_label,
            }
        )
        extra = f", best={best_label}" if rejected else ""
        print(f"{qasm_path} -> {label} (confidence={confidence:.6f}{extra})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
