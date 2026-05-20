"""本地自测：文档加载、样本生成；可选 LangGraph 结构检查。"""
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-local")
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")


def test_loaders():
    from app.rag.loaders import load_directory

    knowledge = BACKEND_ROOT / "data" / "knowledge"
    docs = load_directory(knowledge)
    assert len(docs) >= 6, f"expected >=6 markdown docs, got {len(docs)}"
    types = {d.metadata.get("file_type") for d in docs}
    assert "md" in types
    print(f"[OK] loaders: {len(docs)} documents, types={types}")


def test_graph_structure():
    try:
        from app.graph.workflow import build_graph
    except ImportError as e:
        print(f"[SKIP] graph (install deps): {e}")
        return
    g = build_graph()
    nodes = set(g.get_graph().nodes.keys())
    assert "retrieve" in nodes
    assert "human_select" in nodes
    assert "generate_answer" in nodes
    print(f"[OK] graph nodes: {nodes}")


def test_generate_sample_docs():
    from scripts.generate_sample_docs import main as gen

    gen()
    from app.rag.loaders import load_directory

    knowledge = BACKEND_ROOT / "data" / "knowledge"
    assert (knowledge / "biology_cell_structure.docx").exists()
    assert (knowledge / "math_quadratic_equations.pdf").exists()
    docs = load_directory(knowledge)
    types = {d.metadata.get("file_type") for d in docs}
    assert "docx" in types
    assert "pdf" in types
    print(f"[OK] sample docs: total={len(docs)}, types={types}")


def main():
    test_loaders()
    test_graph_structure()
    test_generate_sample_docs()
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()
