from pathlib import Path
import sys

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    PROJECT_ROOT = Path("/content")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

RAW_DATA_DIR = DATA_DIR / "raw" / "METR-LA"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "METR-LA"

TRAFFIC_FILE = RAW_DATA_DIR / "METR-LA.h5"
ADJACENCY_FILE = RAW_DATA_DIR / "adj_METR-LA.pkl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
