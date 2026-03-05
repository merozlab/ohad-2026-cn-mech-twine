"""
Generate a project overview with folder structure, files, and function definitions.
Creates an overview.txt file in the parent directory showing all folders, files, and functions.
"""
from __future__ import annotations
import os
import ast
import importlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List
import sys


def extract_functions_from_file(filepath: str) -> List[str]:
    """
    Extract function names from a Python file using AST parsing.
    
    Args:
        filepath: Path to the Python file to parse
        
    Returns:
        List of function names found in the file, sorted alphabetically
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return sorted(functions)
    except Exception:
        return []


def get_project_overview(root_dir: str) -> str:
    """
    Generate a comprehensive project overview with directory structure and function listings.
    
    Args:
        root_dir: Root directory of the project to analyze
        
    Returns:
        String containing the formatted project overview
    """
    root_path = Path(root_dir)
    overview = []
    
    overview.append("=" * 80)
    overview.append("PROJECT OVERVIEW")
    overview.append("=" * 80)
    overview.append("")
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip __pycache__ and .git directories
        dirnames[:] = [d for d in dirnames if d not in ['__pycache__', '.git', 'cache']]
        
        # Calculate relative path and indentation
        rel_path = Path(dirpath).relative_to(root_path)
        if str(rel_path) == '.':
            level = 0
            display_path = "ROOT"
        else:
            level = len(rel_path.parts)
            display_path = rel_path.as_posix()
        
        indent = "  " * level
        
        # Add folder header
        overview.append(f"\n{indent}📁 {display_path}/")
        overview.append(f"{indent}{'-' * (len(display_path) + 3)}")
        
        # Filter and sort files
        python_files = sorted([f for f in filenames if f.endswith('.py')])
        other_files = sorted([f for f in filenames if not f.endswith('.py') and not f.startswith('.')])
        
        # Process Python files
        if python_files:
            for filename in python_files:
                filepath = os.path.join(dirpath, filename)
                functions = extract_functions_from_file(filepath)
                
                overview.append(f"{indent}  📄 {filename}")
                if functions:
                    for func in functions:
                        overview.append(f"{indent}      ├─ {func}()")
                else:
                    overview.append(f"{indent}      └─ (no functions)")
        
        # Process other files (non-Python)
        if other_files:
            for filename in other_files:
                overview.append(f"{indent}  📄 {filename}")
    
    return "\n".join(overview)


def print_requirements_file(
    project_root: str | Path = ".",
    output_file: str | Path = "requirements.txt",
    pin_versions: bool = True,
) -> list[str]:
    """
    Scan .py files, infer third-party dependencies, print pinned requirements text,
    and write it to output_file.
    """
    project_root = Path(project_root).resolve()
    output_file = Path(output_file)

    skip_dirs = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "build", "dist"}
    stdlib = set(getattr(sys, "stdlib_module_names", set()))

    pip_name_map = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "mpl_toolkits": "matplotlib",
    }

    def _is_skipped(path: Path) -> bool:
        return any(part in skip_dirs for part in path.parts)

    # Collect local module names (files + package dirs) to exclude
    local_names = {"src"}
    for py in project_root.rglob("*.py"):
        if _is_skipped(py):
            continue
        local_names.add(py.stem)
    for init_py in project_root.rglob("__init__.py"):
        if _is_skipped(init_py):
            continue
        local_names.add(init_py.parent.name)

    imported: set[str] = set()
    for py in project_root.rglob("*.py"):
        if _is_skipped(py):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

    third_party = sorted(
        n for n in imported
        if n and n not in stdlib and n not in local_names
    )

    req_lines: list[str] = []
    unresolved: list[str] = []

    for imp_name in third_party:
        pkg_name = pip_name_map.get(imp_name, imp_name)

        if not pin_versions:
            req_lines.append(pkg_name)
            continue

        ver = None
        for candidate in (pkg_name, imp_name):
            try:
                ver = version(candidate)
                break
            except PackageNotFoundError:
                pass
            except Exception:
                pass

        if ver is None:
            try:
                mod = importlib.import_module(imp_name)
                ver = getattr(mod, "__version__", None)
            except Exception:
                pass

        if ver is None:
            unresolved.append(pkg_name)
            continue

        req_lines.append(f"{pkg_name}=={ver}")

    req_lines = sorted(set(req_lines))
    output_file.write_text("\n".join(req_lines) + ("\n" if req_lines else ""), encoding="utf-8")
    print("\n".join(req_lines))

    if unresolved:
        print(f"[WARN] skipped unresolved/non-distribution imports: {', '.join(sorted(set(unresolved)))}")

    return req_lines


def main():
    """Main entry point: generate overview and save to file."""
    # Get the parent directory (project root)
    root_directory = Path(__file__).parent.parent
    
    # Generate overview
    overview = get_project_overview(str(root_directory))
    
    # Save to overview.txt in the root directory
    output_file = root_directory / "overview.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(overview)
    
    print(f"✓ Overview generated successfully: {output_file}")
    print(f"\nFirst 500 characters:\n{overview[:500]}...\n")

    print_requirements_file(project_root=".", output_file="requirements.txt")

if __name__ == "__main__":
    main()

