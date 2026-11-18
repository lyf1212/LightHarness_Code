"""A limited code contract, not a security sandbox for hostile Python."""
import ast

ALLOWED_IMPORTS = {"bpy", "math", "mathutils", "random"}
FORBIDDEN = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input", "breakpoint"}
BAD_ATTRS = {"load", "save", "save_as_mainfile", "save_mainfile", "write", "render", "quit_blender", "python_file_run", "script", "handlers", "drivers", "driver_add"}


def validate_code(code):
    tree = ast.parse(code)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build_scene" for n in tree.body):
        raise ValueError("scene_code must define build_scene(ctx)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name not in ALLOWED_IMPORTS for a in node.names):
            raise ValueError("Only bpy/math/mathutils/random imports are allowed")
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in ALLOWED_IMPORTS):
            raise ValueError("Unsupported import")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN or node.id.startswith("__")):
            raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BAD_ATTRS):
            raise ValueError(f"Forbidden attribute: {node.attr}")
    return code
