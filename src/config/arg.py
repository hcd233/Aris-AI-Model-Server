from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-cp", type=str, help="Path to model config file, in yaml format", required=True)
    args = vars(parser.parse_args())

    path = Path(args["config_path"])
    if not path.exists():
        raise FileNotFoundError(f"Config file {args['config_path']} not found")

    if path.suffix not in [".yaml", ".yml"]:
        raise ValueError(f"Config file {args['config_path']} must be in yaml format")

    return args




