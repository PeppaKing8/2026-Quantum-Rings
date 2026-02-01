import argparse
import json
from pathlib import Path


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

def predict_with_task(qasm_str: str):
    """
    should return predicted_threshold_min, predicted_forward_wall_s
    """
    pass
    

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

        qasm_text = qasm_path.read_text(encoding="utf-8")

        if wants_task:
            predicted_threshold_min, predicted_forward_wall_s = predict_fn(qasm_text, task)
        else:
            predicted_threshold_min, predicted_forward_wall_s = predict_fn(qasm_text)

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
