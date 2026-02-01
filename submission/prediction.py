from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path


TASK1_DIR = Path(__file__).resolve().parent / "task1"
ALGO_EXP_DIR = TASK1_DIR / "algorithm_good"
ALGO_CONFIG_PATH = TASK1_DIR / "config_algorithm.yaml"
THRESH_MODEL_PATH = TASK1_DIR / "fam_res_reprod_1" / "best_model.pt"
THRESH_CONFIG_PATH = TASK1_DIR / "config.yaml"

RUNTIME_THRESHOLD = 2
ALGO_REJECT_THRESHOLD = 0.7
ALGO_REJECT_LABEL = "other"


def load_tasks(tasks_path: Path):
    with tasks_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported tasks JSON format; expected list or object with key 'tasks'.")


def load_id_map(id_map_path: Path):
    with id_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ID map JSON must include an 'entries' list.")
    id_to_file = {}
    for entry in entries:
        if "id" not in entry or "qasm_file" not in entry:
            raise ValueError("Each ID map entry must include 'id' and 'qasm_file'.")
        id_to_file[entry["id"]] = entry["qasm_file"]
    return id_to_file


def _ensure_task1_on_path() -> None:
    task1_str = str(TASK1_DIR)
    if task1_str not in sys.path:
        sys.path.insert(0, task1_str)


def _auto_device():
    import torch

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


@lru_cache(maxsize=1)
def _load_algorithm_model():
    _ensure_task1_on_path()
    import task1.predict_algorithm as predict_algorithm
    import yaml

    if not ALGO_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing algorithm config: {ALGO_CONFIG_PATH}")

    config = yaml.safe_load(ALGO_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    model_path = ALGO_EXP_DIR / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing algorithm checkpoint: {model_path}")

    feature_stats_path = ALGO_EXP_DIR / "feature_stats.json"
    if not feature_stats_path.exists():
        feature_stats_path = None  # type: ignore[assignment]

    idx_to_label = _load_idx_to_label(ALGO_EXP_DIR)
    device = _auto_device()
    model, registry, _ = predict_algorithm.load_model(
        str(model_path),
        config,
        label_map_path=None,
        device=device,
        feature_stats_path=(str(feature_stats_path) if feature_stats_path is not None else None),
    )
    use_graph = bool(config.get("use_graph_features", True))
    return model, registry, idx_to_label, device, use_graph


def _predict_family_from_qasm(qasm_path: Path) -> str:
    _ensure_task1_on_path()
    import task1.predict_algorithm as predict_algorithm

    model, registry, idx_to_label, device, use_graph = _load_algorithm_model()
    best_label, confidence = predict_algorithm.predict_single_qasm(
        model,
        registry,
        idx_to_label,
        str(qasm_path),
        device,
        use_graph,
    )
    label, _rejected = predict_algorithm.apply_rejection(
        best_label,
        confidence,
        ALGO_REJECT_THRESHOLD,
        ALGO_REJECT_LABEL,
    )
    return _label_to_family(label)


def predict_with_task(qasm_path: str | Path, task: dict):
    """
    Return (predicted_threshold_min, predicted_forward_wall_s) for one task.
    """
    qasm_path = Path(qasm_path)
    if not qasm_path.exists():
        raise FileNotFoundError(f"Missing QASM file: {qasm_path}")

    backend = task.get("processor")
    precision = task.get("precision")
    if backend is None or precision is None:
        raise ValueError("Task must include 'processor' and 'precision'.")

    family = _predict_family_from_qasm(qasm_path)
    disable_family = family == "unknown"

    _ensure_task1_on_path()
    import task1.inference as inference

    preds = inference.predict_threshold_and_runtime(
        str(qasm_path),
        model_path=str(THRESH_MODEL_PATH),
        config_path=str(THRESH_CONFIG_PATH),
        family=family,
        runtime_threshold=RUNTIME_THRESHOLD,
        disable_family=disable_family,
        backend=backend,
        precision=precision,
    )
    if not preds:
        raise RuntimeError(f"No predictions returned for {qasm_path}")

    pred = preds[0]
    return pred.predicted_threshold_min, pred.predicted_forward_wall_s


def resolve_predict_fn():
    return predict_with_task, True


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for holdout tasks.")
    parser.add_argument("--tasks", required=True, help="Path to holdout tasks JSON")
    parser.add_argument("--circuits", required=True, help="Directory containing holdout QASM files")
    parser.add_argument("--id-map", required=True, help="JSON mapping task IDs to QASM files")
    parser.add_argument("--out", required=True, help="Output path for predictions JSON")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    circuits_dir = Path(args.circuits)
    id_map_path = Path(args.id_map)
    out_path = Path(args.out)

    tasks = load_tasks(tasks_path)
    id_to_file = load_id_map(id_map_path)

    predict_fn, wants_task = resolve_predict_fn()

    predictions = []
    for task in tasks:
        task_id = task.get("id")
        if not task_id:
            raise ValueError("Each task must include an 'id'.")
        if task_id not in id_to_file:
            raise KeyError(f"Task id {task_id} missing from ID map.")

        qasm_file = id_to_file[task_id]
        qasm_path = circuits_dir / qasm_file
        if not qasm_path.is_file():
            raise FileNotFoundError(f"Missing QASM file: {qasm_path}")

        if wants_task:
            predicted_threshold_min, predicted_forward_wall_s = predict_fn(qasm_path, task)
        else:
            predicted_threshold_min, predicted_forward_wall_s = predict_fn(qasm_path)

        predictions.append(
            {
                "id": task_id,
                "predicted_threshold_min": int(predicted_threshold_min),
                "predicted_forward_wall_s": float(predicted_forward_wall_s),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


if __name__ == "__main__":
    main()
