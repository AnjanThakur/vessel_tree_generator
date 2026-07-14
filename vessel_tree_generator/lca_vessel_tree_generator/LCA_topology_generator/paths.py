from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def lca_root() -> Path:
    return Path(__file__).resolve().parents[1]


def lca_generated_control_points_dir() -> Path:
    return lca_root() / "LCA_branch_control_points" / "generated"


def outputs_root() -> Path:
    return repo_root() / "outputs"


def output_path(*parts: str) -> Path:
    return outputs_root().joinpath(*parts)
