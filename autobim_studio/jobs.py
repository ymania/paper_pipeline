"""Run an isolated, resumable photo survey in a project directory."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("reconstruct", "segment", "export")
OUTPUT = ROOT / "output" / "studio"


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(job):
    return json.loads((job / "project.json").read_text(encoding="utf-8"))


def save(job, state):
    temp = job / "project.tmp"
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(job / "project.json")


def input_path(value):
    value = str(value).strip().strip('"')
    if len(value) > 2 and value[1:3] in (":/", ":\\"):
        value = subprocess.check_output(["wslpath", "-u", value], text=True).strip()
    return Path(value).expanduser().resolve()


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(source):
    photos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    if len(photos) < 3 or len({p.stem for p in photos}) != len(photos):
        raise ValueError("项目至少需要三张文件主名各异的 JPEG 照片")
    return {p.name: fingerprint(p) for p in photos}


def create(source, model, output=OUTPUT):
    source, model = input_path(source), input_path(model)
    photos = inventory(source)
    model_hash = fingerprint(model)
    job = output / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8])
    job.mkdir(parents=True)
    save(job, dict(source=str(source), model=str(model), photos=photos, model_sha256=model_hash,
                   created=timestamp(), completed=[], status="ready", stage="", error=""))
    return job


def run(job):
    work = job / "work"
    with (job / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(job)
        try:
            if inventory(Path(state["source"])) != state["photos"] or fingerprint(Path(state["model"])) != state["model_sha256"]:
                raise ValueError("输入文件已改变，请创建新项目")
            result = job / "result"
            result.mkdir(exist_ok=True)
            arguments = {
                "reconstruct": [state["source"], work],
                "segment": [work / "images", state["model"]],
                "export": [work, result / "candidate.ifc"]
            }
            for stage in STAGES:
                if stage in state["completed"]:
                    continue
                state.update(stage=stage, status="running", error="")
                save(job, state)
                with (job / "pipeline.log").open("a") as log:
                    print(timestamp(), stage, file=log, flush=True)
                    subprocess.run([sys.executable, "-u", "-m", "autobim_studio."+stage,
                                    *arguments[stage]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                if stage == "export":
                    for extension in ("json", "ifc"):
                        (result / ("candidate."+extension)).replace(result / ("semantic."+extension))
                state["completed"].append(stage)
                save(job, state)
            state.update(status="complete", finished=timestamp())
        except Exception as error:
            state.update(status="failed", error=str(error))
            raise
        finally:
            save(job, state)


def main():
    parser = argparse.ArgumentParser(description="AutoBIM Studio")
    subs = parser.add_subparsers(dest="action", required=True)
    new = subs.add_parser("run")
    new.add_argument("photos")
    new.add_argument("model")
    new.add_argument("--output", type=Path, default=OUTPUT)
    subs.add_parser("resume").add_argument("project", type=Path)
    args = parser.parse_args()
    job = create(args.photos, args.model, args.output) if args.action == "run" else args.project.resolve()
    print(job, flush=True)
    run(job)


if __name__ == "__main__":
    main()
