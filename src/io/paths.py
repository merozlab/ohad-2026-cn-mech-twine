import os
from pathlib import Path

# -------------------------------------------------
# Root directory
# -------------------------------------------------

def get_project_root():
    """
    Project root = folder that contains 'src/'.
    Works from scripts, notebooks, interactive shells.
    Searches parent directories first, then child directories.
    """
    # Search up the directory tree
    for p in Path.cwd().resolve().parents:
        if (p / "src").exists():
            return p
    
    # Search down into child directories
    for p in Path.cwd().resolve().rglob("."):
        if p.is_dir() and (p / "src").exists():
            return p
    
    raise RuntimeError("Could not find project root (folder containing 'src').")

# -------------------------------------------------
# Public paths (single source of truth)
# -------------------------------------------------

# ARTICLE_ROOT = r"C:\Users\Amir\Documents\PHD\Python\GitHub\Amir_Repositories\Repo_article1"
# ARTICLE_ROOT = r"C:\Users\Amir Ohad\Documents\GitHub\Repo_article1"
# DATA_PATH = ARTICLE_ROOT / "Data"
# ARTICLE_ROOT = os.path.abspath(PROJECT_ROOT.parent)
# DATA_PATH = Path(ARTICLE_ROOT) / "Data"

PROJECT_ROOT = get_project_root()

PROJECT_DATA_PATH = PROJECT_ROOT / "data"
CACHE_PATH = PROJECT_DATA_PATH / "cache"

FIGURES_PATH = PROJECT_ROOT / "figures"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

H5_RESULTS_PATH = PROJECT_DATA_PATH / "twine_init_logs" / "2025_analysis"
PEND_TRACK_PATH = PROJECT_DATA_PATH / "track_logs"


# -------------------------------------------------
# Ensure directories exist (safe)
# -------------------------------------------------

def ensure_dirs():
    """
    Create standard project directories if missing.
    """
    for p in [
        PROJECT_DATA_PATH,
        CACHE_PATH,
        FIGURES_PATH,
        OUTPUTS_PATH,
    ]:
        p.mkdir(parents=True, exist_ok=True)
