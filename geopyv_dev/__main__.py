import argparse
import shutil
import sys
from importlib.resources import as_file, files
from pathlib import Path

_TUTORIALS_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".ipynb_checkpoints", "make_tutorials.py"
)


def _cmd_tutorials(args):
    dest = Path(args.dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        print(f"error: destination '{dest}' already exists and is not empty", file=sys.stderr)
        return 1

    with as_file(files("geopyv_dev") / "tutorials") as src:
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=_TUTORIALS_IGNORE)

    print(f"Tutorials copied to {dest}")
    print(f"Run: jupyter notebook {dest}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m geopyv_dev")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tutorials_parser = subparsers.add_parser(
        "tutorials", help="Copy the bundled tutorial notebooks to a local directory"
    )
    tutorials_parser.add_argument(
        "dest",
        nargs="?",
        default="geopyv-dev-tutorials",
        help="Destination directory (default: ./geopyv-dev-tutorials)",
    )
    tutorials_parser.set_defaults(func=_cmd_tutorials)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
