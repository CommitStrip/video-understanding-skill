"""pytest 共享配置。

保证未执行 `pip install -e .` 时，从仓库根直接跑 `python -m pytest tests/`
也能导入 vus 包（把仓库根插入 sys.path，路径用 pathlib，跨 Windows/Linux）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
