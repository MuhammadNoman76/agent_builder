import os

# Folders and files to skip
SKIP_DIRS = {
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    'env',
    '.env',
    '.mypy_cache',
    '.pytest_cache',
    '.vscode',
    '.idea',
    'node_modules',
    'dist',
    'build',
    '.github',
    '.gitignore',  # not a dir but often mistaken
}

SKIP_EXTENSIONS = {
    '.pyc',
    '.pyo',
    '.pyd',
    '.egg',
    '.whl',
    '.md',
    '.txt',
    '.log',
    '.json',
    '.yaml',
    '.yml',
    '.toml',
    '.lock',
    '.env',
}

def get_tree_output():
    """Manually generate a clean tree structure without using shell `tree`."""
    tree_lines = []
    root_dir = os.getcwd()
    tree_lines.append(f"{os.path.basename(root_dir)}/")

    def walk_dir(current_path, prefix=""):
        try:
            items = sorted(os.listdir(current_path))
        except PermissionError:
            return
        files = []
        dirs = []
        for item in items:
            if item in SKIP_DIRS or item.startswith('.') and item not in ('.', '..'):
                # Allow hidden files like .env but skip hidden dirs (like .git, .venv)
                if os.path.isdir(os.path.join(current_path, item)):
                    continue
            full_path = os.path.join(current_path, item)
            if os.path.isdir(full_path):
                if os.path.basename(full_path) not in SKIP_DIRS:
                    dirs.append(item)
            else:
                # Skip unwanted extensions and hidden files (except .env, .gitignore)
                _, ext = os.path.splitext(item)
                if ext.lower() in SKIP_EXTENSIONS and not item.startswith('.'):
                    continue
                if item in SKIP_DIRS:
                    continue
                files.append(item)

        entries = dirs + files
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            tree_lines.append(f"{prefix}{connector}{entry}")
            if entry in dirs:
                extension = "    " if is_last else "│   "
                walk_dir(os.path.join(current_path, entry), prefix + extension)

    walk_dir(root_dir)
    return "\n".join(tree_lines)

def get_python_files(root_dir):
    """Recursively collect all .py files, skipping unwanted directories."""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip unwanted folders
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.') or d in ('.', '..')]
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                # Skip pycache-like files just in case
                if '__pycache__' in full_path:
                    continue
                py_files.append(full_path)
    py_files.sort()
    return py_files

def make_markdown_heading(path, root_dir):
    rel_path = os.path.relpath(path, root_dir).replace(os.sep, '/')
    return f"# {rel_path}"

def main():
    root_dir = os.getcwd()
    output_file = os.path.join(root_dir, "code_documentation.md")

    # Generate clean tree
    tree_output = get_tree_output()

    # Get filtered Python files
    py_files = get_python_files(root_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Project Code Documentation\n\n")
        f.write("## Project Structure\n\n")
        f.write("```\n")
        f.write(tree_output)
        f.write("\n```\n\n")

        for file_path in py_files:
            heading = make_markdown_heading(file_path, root_dir)
            f.write(f"{heading}\n\n")
            f.write("```python\n")
            try:
                with open(file_path, 'r', encoding='utf-8') as pf:
                    f.write(pf.read())
            except Exception as e:
                f.write(f"# ERROR reading file: {e}")
            f.write("\n```\n\n")

    print(f"✅ Clean documentation generated at: {output_file}")

if __name__ == "__main__":
    main()