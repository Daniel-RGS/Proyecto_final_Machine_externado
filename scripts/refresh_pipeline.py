from __future__ import annotations

import argparse
import json
from datetime import datetime

from build_dataset import build_dataset
from collect_acled import collect_acled_data
from collect_firms import collect_firms_data
from collect_gdelt import collect_gdelt_data
from collect_rss import collect_rss_data
from collect_wikimedia import collect_wikimedia_data
from collect_wikipedia_revisions import collect_wikipedia_revisions
from project_paths import DEFAULT_START_DATE, PROCESSED_DIR, default_end_date, ensure_dir
from train_models import train_models


def _frame_rows(frame) -> int:
    return int(len(frame)) if hasattr(frame, "__len__") else 0


def _run_step(name, func, *args, **kwargs):
    print(f"\n=== {name} ===")
    try:
        frame = func(*args, **kwargs)
        rows = _frame_rows(frame)
        print(f"{name}: OK ({rows} filas)")
        return frame, {"status": "ok", "rows": rows}
    except Exception as exc:
        print(f"{name}: ERROR -> {exc}")
        return None, {"status": "error", "error": str(exc)}


def _write_status(status_payload: dict) -> None:
    output_dir = ensure_dir(PROCESSED_DIR)
    status_path = output_dir / "pipeline_status.json"
    with open(status_path, "w", encoding="utf-8") as file_handle:
        json.dump(status_payload, file_handle, indent=2)
    print(f"\nEstado del pipeline guardado en {status_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Actualiza fuentes OSINT, recompone el dataset y reentrena modelos."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=default_end_date())
    parser.add_argument("--include-gdelt", action="store_true")
    parser.add_argument("--include-acled", action="store_true")
    parser.add_argument("--include-firms", action="store_true")
    parser.add_argument("--skip-wikimedia", action="store_true")
    parser.add_argument("--skip-wikipedia", action="store_true")
    parser.add_argument("--skip-rss", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    started_at = datetime.now().isoformat(timespec="seconds")
    sources = {}

    if args.skip_wikimedia:
        sources["wikimedia"] = {"status": "skipped"}
    else:
        _, sources["wikimedia"] = _run_step(
            "Wikimedia Pageviews",
            collect_wikimedia_data,
            args.start_date,
            args.end_date,
        )

    if args.skip_wikipedia:
        sources["wikipedia_revisions"] = {"status": "skipped"}
    else:
        _, sources["wikipedia_revisions"] = _run_step(
            "Wikipedia Revisions",
            collect_wikipedia_revisions,
            args.start_date,
            args.end_date,
        )

    if args.skip_rss:
        sources["rss"] = {"status": "skipped"}
    else:
        _, sources["rss"] = _run_step("RSS", collect_rss_data)

    if args.include_gdelt:
        _, sources["gdelt"] = _run_step(
            "GDELT",
            collect_gdelt_data,
            args.start_date,
            args.end_date,
        )
    else:
        sources["gdelt"] = {"status": "skipped"}

    if args.include_acled:
        _, sources["acled"] = _run_step(
            "ACLED",
            collect_acled_data,
            args.start_date,
            args.end_date,
        )
    else:
        sources["acled"] = {"status": "skipped"}

    if args.include_firms:
        _, sources["firms"] = _run_step(
            "NASA FIRMS",
            collect_firms_data,
            args.start_date,
            args.end_date,
        )
    else:
        sources["firms"] = {"status": "skipped"}

    dataset, dataset_status = _run_step("Build Dataset", build_dataset)
    model_metadata = None
    model_status = {"status": "skipped"} if args.skip_train else None
    if not args.skip_train:
        model_metadata, model_status = _run_step("Train Models", train_models)

    dataset_summary = None
    if dataset is not None and not getattr(dataset, "empty", True):
        dataset_summary = {
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns)),
            "date_min": dataset["date"].min().strftime("%Y-%m-%d"),
            "date_max": dataset["date"].max().strftime("%Y-%m-%d"),
        }

    status_payload = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "sources": sources,
        "dataset": dataset_status,
        "dataset_summary": dataset_summary,
        "model": model_status,
        "model_summary": model_metadata,
    }
    _write_status(status_payload)


if __name__ == "__main__":
    main()
