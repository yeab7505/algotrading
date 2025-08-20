import os
import sys


def main() -> None:
	# Ensure project root is importable (so `tools.py` and others can be found)
	project_root = os.path.dirname(__file__)
	if project_root not in sys.path:
		sys.path.insert(0, project_root)

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


