"""Shared helpers for the profiling scripts: path setup and output folder."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "results" / "profiling"
OUT.mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_checked(cmd, cwd):
    """Run cmd silently; on failure, raise with the command's stderr."""
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed (exit {out.returncode}):\n"
                           f"{out.stderr}")
    return out


def done(script):
    """Report that a profiling script finished and where its output is."""
    print(f"{Path(script).name}: done, outputs in {OUT.relative_to(ROOT)}/")
