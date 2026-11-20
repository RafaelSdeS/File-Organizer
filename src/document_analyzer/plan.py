"""A run's proposed folders as an editable text file (--plan / --apply), and
the checks any user-edited plan (file or --ui) must pass before files move.

    # organize-files plan	source=/abs/src	target=/abs/dst
    [Solar_system]
    0.41	lua.txt
    0.83	mars.txt

Rename a [folder], move a line to move a file, delete a line to leave that
file where it is. The number (how much a file resembles its folder) is
optional and ignored when reading.
"""
import os

HEADER = "# organize-files plan"


def write_plan(path, folders, source, target, scores=None, sensitive=None):
    scores, sensitive = scores or {}, sensitive or {}
    lines = [
        f"{HEADER}\tsource={os.path.abspath(source)}\ttarget={os.path.abspath(target)}",
        "# Rename a [folder], move a file's line to another folder, or delete it to leave the file where it is.",
        "# The number is how much a file resembles its folder; the lowest come first in each folder.",
    ]
    for folder, files in folders.items():
        lines.append(f"\n[{folder}]")
        for name in sorted(files, key=lambda f: scores.get(f, 1.0)):
            if name in sensitive:
                lines.append(f"# sensitive: {', '.join(sensitive[name])}")
            lines.append(f"{scores[name]:.2f}\t{name}" if name in scores else name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_plan(path):
    """(folders, source, target) from a plan file; ValueError if it isn't one."""
    folders, source, target, current = {}, None, None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith(HEADER):
                fields = dict(part.split("=", 1) for part in line.split("\t")[1:] if "=" in part)
                source, target = fields.get("source"), fields.get("target")
            elif not line.strip() or line.startswith("#"):
                continue
            elif line.startswith("[") and line.rstrip().endswith("]"):
                current = line.strip()[1:-1].strip()
                folders.setdefault(current, [])
            elif current is None:
                raise ValueError(f"File listed before any [folder]: {line!r}")
            else:
                score, tab, name = line.partition("\t")
                folders[current].append(name if tab and _is_number(score) else line)
    if not source:
        raise ValueError(f"{path} is not a plan file (no '{HEADER}' header line)")
    return {folder: files for folder, files in folders.items() if files}, source, target or source


def validate_folders(folders, source):
    """Raise ValueError unless every folder is a relative path that stays
    inside the target ("Parent/Child" allowed) and every file is an entry
    directly inside source, listed once. Plans are typed by hand (or come
    from the UI), so this is the trust boundary before anything moves."""
    seen = set()
    for folder, files in folders.items():
        parts = folder.split("/")
        if "\\" in folder or ":" in folder or any(part.strip() in ("", ".", "..") for part in parts):
            raise ValueError(f"Invalid folder name: {folder!r}")
        for name in files:
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError(f"Invalid file name: {name!r}")
            if name in seen:
                raise ValueError(f"{name!r} is listed in more than one folder")
            seen.add(name)
            if not os.path.lexists(os.path.join(source, name)):
                raise ValueError(f"{name!r} is not in {source}")


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False
