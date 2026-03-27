import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import (
    append_tsv_row,
    ensure_dir,
    load_config,
    resolve_path,
    run_command,
    try_git_commit,
    utc_timestamp,
    write_json,
)
from evaluate_hf import evaluate_model as evaluate_hf_model
from evaluate_sherpa import evaluate_model as evaluate_sherpa_model
from merge_lora import merge_lora
from prepare_eval_audio import build_eval_manifest
from quantize_onnx import quantize_directory


def maybe_run_export_hook(config, config_dir: Path, repo_root: Path):
    hook_command = config.get("export", {}).get("hook_command", "").strip()
    if not hook_command:
        return {
            "status": "skipped",
            "reason": "export.hook_command is empty",
        }

    merged_model_dir = resolve_path(config_dir, config["paths"]["merged_model_dir"])
    onnx_fp32_dir = ensure_dir(resolve_path(config_dir, config["paths"]["onnx_fp32_dir"]))
    workspace_root = config_dir
    base_model_name = config["base_model_name"]

    formatted = hook_command.format(
        base_model_name=base_model_name,
        merged_model_dir=str(merged_model_dir),
        onnx_fp32_dir=str(onnx_fp32_dir),
        repo_root=str(repo_root),
        workspace_root=str(workspace_root),
    )
    result = run_command(formatted, cwd=repo_root)
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "command": formatted,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def format_metric(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Run the STT autoresearch evaluation pipeline.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--description", default="manual run")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config, config_dir = load_config(config_path)
    repo_root = config_dir.parent
    paths = config["paths"]

    run_id = utc_timestamp()
    run_summaries_dir = ensure_dir(resolve_path(config_dir, paths["run_summaries_dir"]))
    summary_path = run_summaries_dir / f"{run_id}.json"
    results_tsv = resolve_path(config_dir, paths["results_tsv"])
    git_commit = try_git_commit(repo_root)

    summary = {
        "run_id": run_id,
        "git_commit": git_commit,
        "description": args.description,
    }

    try:
        summary["prepare_eval_audio"] = build_eval_manifest(config_path)
        summary["merge_lora"] = merge_lora(config_path)

        merged_model_dir = resolve_path(config_dir, paths["merged_model_dir"])
        hf_result = evaluate_hf_model(config_path, merged_model_dir)
        summary["hf_merged_eval"] = hf_result

        export_result = maybe_run_export_hook(config, config_dir, repo_root)
        summary["export"] = export_result

        fp32_dir = resolve_path(config_dir, paths["onnx_fp32_dir"])
        int8_dir = resolve_path(config_dir, paths["onnx_int8_dir"])
        sherpa_fp32 = None
        sherpa_int8 = None

        if export_result["status"] == "ok" and any(fp32_dir.glob("*.onnx")):
            sherpa_fp32 = evaluate_sherpa_model(config_path, fp32_dir)
            summary["sherpa_fp32_eval"] = sherpa_fp32

            summary["quantize"] = quantize_directory(fp32_dir, int8_dir)
            if any(int8_dir.glob("*.onnx")):
                sherpa_int8 = evaluate_sherpa_model(config_path, int8_dir)
                summary["sherpa_int8_eval"] = sherpa_int8

        status = "ok" if sherpa_int8 is not None else "partial"
        summary["status"] = status

        write_json(summary_path, summary)
        append_tsv_row(
            results_tsv,
            [
                git_commit,
                format_metric(hf_result.get("wer")),
                format_metric(hf_result.get("war")),
                format_metric(None if sherpa_fp32 is None else sherpa_fp32.get("wer")),
                format_metric(None if sherpa_int8 is None else sherpa_int8.get("wer")),
                status,
                args.description,
            ],
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        write_json(summary_path, summary)
        append_tsv_row(
            results_tsv,
            [git_commit, "", "", "", "", "failed", f"{args.description} | {exc!r}"],
        )
        raise


if __name__ == "__main__":
    main()
