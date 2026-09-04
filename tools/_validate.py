"""Quick structural validation for the generated notebooks & python twins."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 1) notebooks must be valid nbformat-4 JSON
nbs = sorted(ROOT.glob("[0-9]*/*.ipynb"))
print(f"Found {len(nbs)} notebooks")
for nb in nbs:
    data = json.loads(nb.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4, nb
    cells = data["cells"]
    kinds = [c["cell_type"] for c in cells]
    assert all(k in ("markdown", "code") for k in kinds), nb
    src_ok = all(isinstance(line, str) for c in cells for line in c["source"])
    print(
        f"  OK  {str(nb.relative_to(ROOT)):60s} "
        f"cells={len(cells):2d} md={kinds.count('markdown'):2d} "
        f"code={kinds.count('code'):2d} source_ok={src_ok}"
    )
    assert src_ok, nb

# 2) python twins must compile
pys = sorted(ROOT.glob("[0-9]*/*.py"))
print(f"\nCompile-checking {len(pys)} python twins")
for py in pys:
    compile(py.read_text(encoding="utf-8"), str(py), "exec")
    print(f"  OK  {py.name}")
print("\nAll good.")
