import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def load_config(config_path: str | os.PathLike):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config, config_path.parent


def resolve_path(base_dir: Path, value: str | os.PathLike) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(path: Path, payload):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_tsv_row(path: Path, columns: list[str]):
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write("\t".join(columns) + "\n")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def try_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_command(command: str, cwd: Path):
    return subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
    )
