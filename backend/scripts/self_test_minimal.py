"""无第三方 AI 依赖的结构自测。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ROOT / "data" / "knowledge"

MD_FILES = [
    "cs_python_basics.md",
    "history_ancient_china.md",
    "chemistry_periodic_table.md",
    "geography_climate_zones.md",
    "literature_poetry_analysis.md",
    "math_calculus_intro.md",
]

REQUIRED_PATHS = [
    ROOT / "app" / "main.py",
    ROOT / "app" / "graph" / "workflow.py",
    ROOT / "app" / "rag" / "vector_store.py",
    ROOT / "app" / "scheduler" / "daily_push.py",
    ROOT.parent / "frontend" / "src" / "App.tsx",
]


def main():
    for p in REQUIRED_PATHS:
        assert p.exists(), f"missing: {p}"
    print(f"[OK] project structure ({len(REQUIRED_PATHS)} paths)")

    for name in MD_FILES:
        path = KNOWLEDGE / name
        assert path.exists(), f"missing markdown: {name}"
        assert path.stat().st_size > 200, f"too small: {name}"
    print(f"[OK] markdown knowledge base ({len(MD_FILES)} files)")

    print("\nMinimal self-test passed. Run full test after: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
