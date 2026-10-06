"""
mi_analysis/src/fingerprint.py — SHA-256 fingerprints, manifests and the experiment registry (MB01).

A SHA-256 fingerprint is a 64-character code computed from a file's bytes; changing one byte
changes it. Recording fingerprints lets every later notebook prove it read exactly the files
that an earlier notebook wrote.
"""
import os
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def rowset_sha256(identifiers):
    """Fingerprint of a SET of row identifiers, independent of row order."""
    text = ",".join(sorted(str(value) for value in identifiers))
    return hashlib.sha256(text.encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        with open(manifest_path) as handle:
            return json.load(handle)
    return {}


def record_files(manifest_path, named_paths):
    """Add or refresh entries {name: {path, bytes, sha256, recorded_at}} in a manifest file."""
    manifest = load_manifest(manifest_path)
    for name, path in named_paths.items():
        path = Path(path)
        manifest[name] = {"path": os.path.relpath(path.resolve(), Path(manifest_path).resolve().parents[1]), "bytes": path.stat().st_size,
                          "sha256": file_sha256(path), "recorded_at": now()}
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def record_value(manifest_path, name, sha256, note=""):
    """Record a fingerprint that is not a file (for example a row-set fingerprint)."""
    manifest = load_manifest(manifest_path)
    manifest[name] = {"path": None, "bytes": None, "sha256": sha256, "note": note, "recorded_at": now()}
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def verify_files(manifest_path):
    """Re-hash every file-backed entry in the manifest; raise if any no longer matches.

    Later notebooks call this before reading anything an earlier one wrote, so a changed or
    regenerated file is caught instead of silently feeding a stale analysis. Row-set entries have
    no path and are skipped.
    """
    manifest = load_manifest(manifest_path)
    verified = []
    for name, entry in manifest.items():
        path = entry.get("path")
        if not path:
            continue
        path = Path(path) if Path(path).is_absolute() else Path(manifest_path).resolve().parents[1] / path
        if not Path(path).exists():
            raise RuntimeError(f"BLOCKED: '{name}' recorded at {path} no longer exists.")
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"BLOCKED: '{name}' at {path} has changed since it was recorded "
                               f"(expected {entry['sha256'][:16]}..., got {actual[:16]}...).")
        verified.append(name)
    return verified


def snapshot_folders(folders, patterns=("*.py", "*.json", "*.csv")):
    """{relative path: sha256} for every matching file under the given folders."""
    snapshot = {}
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for pattern in patterns:
            for path in sorted(folder.rglob(pattern)):
                snapshot[str(path)] = file_sha256(path)
    return snapshot


def register(registry_path, entry):
    """Append one experiment record (one JSON line) to the experiment registry."""
    entry = dict(entry)
    entry["time"] = now()
    with open(registry_path, "a") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def write_snapshot(snapshot_path, folders, patterns=("*.py", "*.json", "*.csv")):
    """Record {path: sha256} for every matching file under the given folders."""
    snapshot = snapshot_folders(folders, patterns)
    with open(snapshot_path, "w") as handle:
        json.dump({"recorded_at": now(), "files": snapshot}, handle, indent=2)
    return snapshot


def compare_snapshot(snapshot_path):
    """Re-hash a snapshot and return the files that no longer match; empty means nothing changed."""
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.exists():
        raise RuntimeError(f"BLOCKED: {snapshot_path} does not exist, so no integrity claim can be made.")
    with open(snapshot_path) as handle:
        recorded = json.load(handle)
    files = recorded.get("files", recorded)
    changed = []
    for path, digest in files.items():
        if not Path(path).exists():
            changed.append({"path": path, "reason": "missing"})
        elif file_sha256(path) != digest:
            changed.append({"path": path, "reason": "changed"})
    return changed
