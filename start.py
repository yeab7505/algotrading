import os
import sys
import importlib.util


def main() -> None:
	# Ensure project root is importable (so `tools.py` and others can be found)
	project_root = os.path.dirname(__file__)
	if project_root not in sys.path:
		sys.path.insert(0, project_root)

	# Preload tools.py as module 'tools' if present to avoid import errors
	tools_path = os.path.join(project_root, "tools.py")
	if os.path.exists(tools_path) and "tools" not in sys.modules:
		spec = importlib.util.spec_from_file_location("tools", tools_path)
		if spec and spec.loader:
			module = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(module)
			sys.modules["tools"] = module

	# Resolve the path to the target script with a space in the filename
	script_path = os.path.join(project_root, "ichimoku live.py")
	if not os.path.exists(script_path):
		sys.stderr.write(f"Start error: script not found at {script_path}\n")
		sys.exit(1)

	# Execute the target script as __main__
	globals_dict = {"__name__": "__main__", "__file__": script_path}
	with open(script_path, "rb") as f:
		code = compile(f.read(), script_path, "exec")
	exec(code, globals_dict)


if __name__ == "__main__":
	main()


