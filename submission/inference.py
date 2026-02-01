"""
Lightweight inference helpers for the iQuHACK 2026 model.

Main entrypoint:
    predict_threshold_and_runtime(qasm_file_name, ...)

Given a QASM filename (or path), this returns the model's predicted threshold
and runtime for one or more (backend, precision) configurations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataset import FAMILY_TO_IDX, THRESHOLD_LADDER, THRESHOLD_TO_IDX
from src.features import qasm_file_to_features
from src.feature_extractors import (
    create_default_registry,
    create_extended_registry,
    create_minimal_registry,
)
from src.model import create_model


@dataclass(frozen=True)
class Prediction:
    file: str
    backend: str
    precision: str
    predicted_threshold_min: int
    predicted_forward_wall_s: float


def _normalize_family_name(name: str) -> str:
    return name.strip().replace("-", "_").replace(" ", "_").lower()


def _family_idx_from_name(family: str | None) -> int:
    if family is None:
        family = "unknown"
    norm = _normalize_family_name(family)
    if norm in ("", "unknown", "unk", "none", "null"):
        return max(FAMILY_TO_IDX.values(), default=0) + 1

    normalized_map = {_normalize_family_name(k): v for k, v in FAMILY_TO_IDX.items()}
    if norm in normalized_map:
        return int(normalized_map[norm])
    raise ValueError(
        f"Unknown family '{family}'. Expected one of: {sorted(FAMILY_TO_IDX.keys())} or 'unknown'."
    )


def _auto_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_registry(name: str):
    if name == "default":
        return create_default_registry()
    if name == "minimal":
        return create_minimal_registry()
    if name == "extended":
        return create_extended_registry()
    raise ValueError(f"Unknown feature_registry: {name}")


def _resolve_qasm_path(qasm_file_name: str, circuits_dir: str | Path) -> Path:
    circuits_dir = Path(circuits_dir)
    candidate = Path(qasm_file_name)
    if candidate.exists():
        return candidate

    name = qasm_file_name
    if not name.endswith(".qasm"):
        name = f"{name}.qasm"

    direct = circuits_dir / name
    if direct.exists():
        return direct

    # Fallback: search within circuits_dir (covers e.g. circuits/original_bench/*).
    matches = list(circuits_dir.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find QASM file '{qasm_file_name}' under {circuits_dir}")


def _default_feature_stats_path(model_path: str | Path) -> Path | None:
    model_path = Path(model_path)
    candidate = model_path.parent / "feature_stats.json"
    if candidate.exists():
        return candidate
    fallback = Path("exp/default/feature_stats.json")
    if fallback.exists():
        return fallback
    return None


@lru_cache(maxsize=2)
def _load_model_and_registry(
    model_path: str,
    config_path: str | None,
    feature_stats_path: str | None,
    device_str: str | None,
) -> tuple[torch.nn.Module, Any, dict[str, Any], torch.device]:
    device = _auto_device(device_str)

    checkpoint = torch.load(model_path, map_location=device)
    if config_path is not None:
        import yaml

        with Path(config_path).open("r", encoding="utf-8") as f:
            config = dict(yaml.safe_load(f) or {})
    else:
        config = dict(checkpoint.get("config", {}))

    registry_name = config.get("feature_registry", "default")
    registry = _create_registry(registry_name)

    stats_path = Path(feature_stats_path) if feature_stats_path is not None else _default_feature_stats_path(model_path)
    if stats_path is not None and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is not None and std is not None:
            registry.set_standardization(mean, std)

    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, registry, config, device


def predict_threshold_and_runtime(
    qasm_file_name: str,
    *,
    circuits_dir: str | Path = "circuits",
    model_path: str = "exp/fam_res_reprod_1/best_model.pt",
    config_path: str | None = None,
    feature_stats_path: str | None = None,
    device: str | None = None,
    family: str | None = None,
    runtime_threshold: int = 16,
    disable_family: bool = False,
    backend: str | None = None,
    precision: str | None = None,
) -> list[Prediction]:
    """
    Predict (threshold, runtime) for the given circuit.

    Args:
        qasm_file_name: QASM filename (e.g. 'qft_indep_qiskit_15.qasm') or path.
        circuits_dir: Directory containing QASM files.
        model_path: Checkpoint path (expects keys: model_state_dict, config).
        feature_stats_path: Optional feature_stats.json for standardization.
        device: Optional torch device string (e.g. 'cpu', 'cuda').
        family: Circuit family name (e.g. 'GHZ'). Use 'unknown' if not sure.
        runtime_threshold: Threshold value to feed to the runtime head when
            the model uses threshold as an input (must be one of THRESHOLD_LADDER).
        disable_family: If True, zero out family features and skip any FiLM/residual
            family modulation inside the model (requires model support).
        backend: Optional filter ('CPU' or 'GPU').
        precision: Optional filter ('single' or 'double').

    Returns:
        List of predictions, one per requested (backend, precision).
    """
    from torch_geometric.data import Batch, Data

    model, registry, config, torch_device = _load_model_and_registry(
        model_path=str(model_path),
        config_path=config_path,
        feature_stats_path=feature_stats_path,
        device_str=device,
    )
    setattr(model, "disable_family", bool(disable_family))

    qasm_path = _resolve_qasm_path(qasm_file_name, circuits_dir)
    family_idx_val = _family_idx_from_name(family)

    use_graph = bool(config.get("use_graph_features", True))
    use_threshold_input = bool(config.get("use_threshold_input", False))

    if runtime_threshold not in THRESHOLD_TO_IDX:
        raise ValueError(
            f"runtime_threshold must be one of {THRESHOLD_LADDER}, got {runtime_threshold!r}"
        )
    runtime_threshold_idx = torch.tensor(THRESHOLD_TO_IDX[runtime_threshold], dtype=torch.long)

    requested: list[tuple[str, str]] = []
    if backend is not None and precision is not None:
        requested = [(backend, precision)]
    elif backend is not None:
        requested = [(backend, "double"), (backend, "single")]
    elif precision is not None:
        requested = [("CPU", precision), ("GPU", precision)]
    else:
        requested = [("CPU", "double"), ("CPU", "single"), ("GPU", "double"), ("GPU", "single")]

    out: list[Prediction] = []
    file_name = Path(qasm_path).name
    for backend_val, precision_val in requested:
        backend_bit = torch.tensor(1.0 if backend_val == "GPU" else 0.0, dtype=torch.float32)
        precision_bit = torch.tensor(1.0 if precision_val == "single" else 0.0, dtype=torch.float32)
        circuit_family_idx = torch.tensor(family_idx_val, dtype=torch.long)

        features = qasm_file_to_features(
            qasm_path,
            backend=backend_val,
            precision=precision_val,
            registry=registry,
            augment=False,
        )

        data_kwargs: dict[str, Any] = {
            "global_features": features["global_features"],
            "backend_bit": backend_bit,
            "precision_bit": precision_bit,
            "circuit_family_idx": circuit_family_idx,
        }
        if use_threshold_input:
            data_kwargs["input_threshold_idx"] = runtime_threshold_idx

        if use_graph:
            num_nodes = int(len(features["gate_type_idx"]))
            data_kwargs.update(
                {
                    "num_nodes": num_nodes,
                    "gate_type_idx": features["gate_type_idx"],
                    "qubit_indices": features["qubit_indices"],
                    "is_directional": features["is_directional"],
                    "gate_arity": features["gate_arity"],
                    "gate_index_norm": features["gate_index_norm"],
                    "edge_index": features["edge_index"],
                }
            )

        data = Data(**data_kwargs)
        batch = Batch.from_data_list([data]).to(torch_device)

        with torch.no_grad():
            threshold_logits, log_runtime = model(batch)

        threshold_class = int(threshold_logits.argmax(dim=1).item())
        predicted_threshold = 256 if threshold_class == 9 else int(THRESHOLD_LADDER[threshold_class])
        predicted_runtime = float(np.exp(float(log_runtime.item())))

        out.append(
            Prediction(
                file=file_name,
                backend=backend_val,
                precision=precision_val,
                predicted_threshold_min=predicted_threshold,
                predicted_forward_wall_s=predicted_runtime,
            )
        )

    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict threshold and runtime for a QASM circuit")
    parser.add_argument(
        "--qasm",
        type=str,
        required=True,
        help="QASM filename or a path to a .qasm file",
    )
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        help="Circuit family name (e.g. 'GHZ'). Use 'unknown' if not sure.",
    )
    parser.add_argument(
        "--runtime-threshold",
        type=int,
        default=16,
        choices=THRESHOLD_LADDER,
        help="Threshold value used as input to the runtime head (when enabled in the model config).",
    )
    parser.add_argument(
        "--circuits-dir",
        type=str,
        default="circuits",
        help="Directory containing the QASM files (if --qasm is not a full path)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional model config YAML; if omitted, uses the config saved in the checkpoint",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="exp/fam_res_reprod_1/best_model.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--feature-stats",
        type=str,
        default=None,
        help="Optional feature_stats.json path (for global feature standardization)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cpu, cuda). Default: auto-detect",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["CPU", "GPU"],
        help="Optional backend filter (CPU/GPU)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["single", "double"],
        help="Optional precision filter (single/double)",
    )

    args = parser.parse_args(argv)

    preds = predict_threshold_and_runtime(
        args.qasm,
        circuits_dir=args.circuits_dir,
        config_path=args.config,
        model_path=args.model_path,
        feature_stats_path=args.feature_stats,
        device=args.device,
        family=args.family,
        runtime_threshold=args.runtime_threshold,
        backend=args.backend,
        precision=args.precision,
    )

    for p in preds:
        print(
            f"{p.file} ({p.backend}, {p.precision}) -> "
            f"threshold={p.predicted_threshold_min}, runtime_s={p.predicted_forward_wall_s:.6f}"
        )


if __name__ == "__main__":
    main()
