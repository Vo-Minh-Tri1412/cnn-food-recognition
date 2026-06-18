from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from canteen_checkout.config import DEFAULT_DETECTOR_PATH, DETECTION_DIR, OUTPUTS_DIR, PROJECT_ROOT, REPORTS_DIR
from canteen_checkout.yolo_runtime import resolve_yolo_model_reference, yolo_cache_working_directory


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the YOLO egg/fish detector with Ultralytics.")
    parser.add_argument("--data", type=Path, default=DETECTION_DIR / "egg_fish" / "data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--project", type=Path, default=OUTPUTS_DIR / "yolo_runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights-cache", type=Path, default=OUTPUTS_DIR / "cache" / "ultralytics")
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else (PROJECT_ROOT / args.data).resolve()
    model_out = args.model_out if args.model_out.is_absolute() else (PROJECT_ROOT / args.model_out).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_path}")

    from ultralytics import YOLO

    run_name = args.name or f"egg_fish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_path = args.project if args.project.is_absolute() else PROJECT_ROOT / args.project
    model_reference = resolve_yolo_model_reference(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "project": str(project_path.resolve()),
        "name": run_name,
        "workers": args.workers,
        "seed": args.seed,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    with yolo_cache_working_directory(args.weights_cache):
        model = YOLO(model_reference)
        results = model.train(**train_kwargs)
        save_dir = Path(results.save_dir)
        best_path = save_dir / "weights" / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Training finished but best.pt was not found: {best_path}")

        model_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, model_out)
        try:
            metrics = YOLO(str(model_out)).val(
                data=str(data_path),
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                project=str(project_path.resolve()),
                name=f"{run_name}_val",
                exist_ok=True,
            )
            metrics_payload = getattr(metrics, "results_dict", {})
        except Exception as exc:
            metrics_payload = {"val_error": f"{type(exc).__name__}: {exc}"}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data": relative_or_absolute(data_path),
        "base_model": args.model,
        "weights_cache": relative_or_absolute(args.weights_cache),
        "run_dir": relative_or_absolute(save_dir),
        "best_path": relative_or_absolute(best_path),
        "model_out": relative_or_absolute(model_out),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "metrics": metrics_payload,
    }
    summary_path = REPORTS_DIR / "egg_fish_detector_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
