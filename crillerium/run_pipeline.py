from pathlib import Path
import os

project_root = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str((project_root / ".mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from awa_pipeline.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(project_root=project_root)
