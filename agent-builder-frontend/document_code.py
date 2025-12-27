import os

# Folders to skip
SKIP_DIRS = {
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    'env',
    '.mypy_cache',
    '.pytest_cache',
    '.vscode',
    '.idea',
    'node_modules',
    'dist',
    'build',
    '.github',
    '.next',
    'out',
    'coverage',
    '.expo',
    'android',
    'ios',
    "document_code.py",
}

# File extensions to include (React-focused)
INCLUDE_EXTENSIONS = {
    '.ts',
    '.tsx',
    '.js',
    '.jsx',
    '.css',
    '.scss',
    '.sass',
    '.json',
    '.md',
    '.html',
    '.yaml',
    '.yml',
    '.toml',
    '.env',
    '.env.local',
    'package-lock.json',
}

# Map extensions to Markdown language identifiers
EXTENSION_TO_LANG = {
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.js': 'javascript',
    '.jsx': 'jsx',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'sass',
    '.json': 'json',
    '.md': 'markdown',
    '.html': 'html',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.env': 'ini',
    '.env.local': 'ini',
}

def get_tree_output():
    """Manually generate a clean tree structure."""
    tree_lines = []
    root_dir = os.getcwd()
    tree_lines.append(f"{os.path.basename(root_dir)}/")

    def walk_dir(current_path, prefix=""):
        try:
            items = sorted(os.listdir(current_path))
        except (PermissionError, OSError):
            return
        files = []
        dirs = []
        for item in items:
            full_path = os.path.join(current_path, item)
            is_dir = os.path.isdir(full_path)

            # Skip unwanted directories
            if is_dir:
                if item in SKIP_DIRS or (item.startswith('.') and item not in ('.', '..')):
                    continue
                dirs.append(item)
            else:
                # Include only specified file types
                _, ext = os.path.splitext(item)
                if ext.lower() in INCLUDE_EXTENSIONS or item in ('.env', '.env.local'):
                    files.append(item)
                elif item in ('package.json', 'tsconfig.json', 'tailwind.config.js', 'postcss.config.js', 'vite.config.js', 'next.config.js'):
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

def get_frontend_files(root_dir):
    """Collect all relevant frontend files."""
    frontend_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip unwanted directories in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (d.startswith('.') and d not in ('.', '..'))]

        for file in files:
            full_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            # Include by extension or known config filename
            if (ext.lower() in INCLUDE_EXTENSIONS or
                file in ('package.json', 'tsconfig.json', 'tailwind.config.js', 'postcss.config.js', 'vite.config.js', 'next.config.js') or
                file in ('.env', '.env.local')):
                frontend_files.append(full_path)
    frontend_files.sort()
    return frontend_files

def get_code_language(filepath):
    """Return the appropriate Markdown language for syntax highlighting."""
    basename = os.path.basename(filepath)
    if basename in ('package.json', 'tsconfig.json'):
        return 'json'
    if basename in ('tailwind.config.js', 'postcss.config.js', 'vite.config.js', 'next.config.js'):
        return 'javascript'
    if basename in ('.env', '.env.local'):
        return 'ini'

    _, ext = os.path.splitext(filepath)
    return EXTENSION_TO_LANG.get(ext.lower(), '')

def make_markdown_heading(path, root_dir):
    rel_path = os.path.relpath(path, root_dir).replace(os.sep, '/')
    return f"# {rel_path}"

def main():
    root_dir = os.getcwd()
    output_file = os.path.join(root_dir, "react_documentation.md")

    # Generate tree
    tree_output = get_tree_output()

    # Get relevant files
    frontend_files = get_frontend_files(root_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# React Application Documentation\n\n")
        f.write("## Project Structure\n\n")
        f.write("```text\n")
        f.write(tree_output)
        f.write("\n```\n\n")

        for file_path in frontend_files:
            heading = make_markdown_heading(file_path, root_dir)
            lang = get_code_language(file_path)
            f.write(f"{heading}\n\n")
            f.write(f"```{lang}\n")
            try:
                with open(file_path, 'r', encoding='utf-8') as pf:
                    f.write(pf.read())
            except Exception as e:
                f.write(f"<!-- ERROR reading file: {e} -->")
            f.write("\n```\n\n")

    print(f"✅ React documentation generated at: {output_file}")

if __name__ == "__main__":
    main()