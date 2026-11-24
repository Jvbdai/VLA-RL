"""Capture the actual source, dependency, and model identity for each run."""
import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

def provenance(root, checkpoint=None):
    root = Path(root)
    def git(path, *args):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ("src", "scripts") for p in (root / folder).rglob("*.py")}
    packages = {}
    for name in ("torch", "mujoco", "gymnasium", "stable-baselines3", "lerobot", "robocasa", "robosuite"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"git_head": git(root, "rev-parse", "HEAD"), "git_status": git(root, "status", "--short"),
            "dependency_heads": {p: git(root / p, "rev-parse", "HEAD") for p in ("robocasa", "robosuite", "third_party/lerobot")},
            "source_sha256": source_hashes, "packages": packages,
            "checkpoint": str(checkpoint) if checkpoint else None}
