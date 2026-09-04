"""Per-cell syntax check: compile every code cell of every notebook."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
errors = []
total = 0
for nb in sorted(ROOT.glob("[0-9]*/*.ipynb")):
    data = json.loads(nb.read_text(encoding="utf-8"))
    for i, cell in enumerate(data["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Strip notebook-only shell lines before the syntax check
        clean = "\n".join(
            ln for ln in src.split("\n")
            if not ln.strip().startswith("!") and not ln.strip().startswith("%")
        )
        total += 1
        try:
            compile(clean, f"{nb.name}:cell{i}", "exec")
        except SyntaxError as err:
            errors.append(f"{nb.name} cell {i}: {err}")
if errors:
    print("SYNTAX ERRORS FOUND:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)
print(f"All {total} code cells across 4 notebooks compile cleanly.")
