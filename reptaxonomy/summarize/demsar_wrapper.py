from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reptaxonomy.util.general_utils import read_yaml
from demsar_rank import run_demsar


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def aggregate_mean_std(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    mean_name: str,
    std_name: str,
) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": mean_name, "std": std_name, "count": "n_runs"})
    )
    out[std_name] = out[std_name].fillna(0.0)
    return out


def resolve_dataset_name(cfg: dict[str, Any]) -> str:
    dataset_bundle = cfg.get("dataset_bundle")
    if isinstance(dataset_bundle, dict):
        dataset_spec = dataset_bundle.get("dataset_spec", {})
        if isinstance(dataset_spec, dict):
            name = dataset_spec.get("dataset_name") or dataset_spec.get("name")
            if name:
                return str(name)

    dataset_cfg = cfg.get("dataset")
    if isinstance(dataset_cfg, dict):
        name = dataset_cfg.get("dataset_name") or dataset_cfg.get("name")
        if name:
            return str(name)

    if isinstance(dataset_cfg, str):
        return dataset_cfg

    raise ValueError("Could not resolve dataset name from cfg['dataset_bundle'] or cfg['dataset'].")


def resolve_model_names(cfg: dict[str, Any]) -> list[str]:
    model_names_cfg = cfg.get("model_names")
    if isinstance(model_names_cfg, (str, Path)):
        model_names = load_json(Path(model_names_cfg))
    elif isinstance(model_names_cfg, list):
        model_names = model_names_cfg
    else:
        raise ValueError("model_names must be either a list or a path to a JSON file")

    if not isinstance(model_names, list) or not model_names:
        raise ValueError("model_names must resolve to a non-empty list")

    return [str(x) for x in model_names]


def build_silhouette_inputs(
    output_dir: Path,
    dataset_name: str,
    model_names: list[str],
    demsar_input_dir: Path,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[Path]]:
    metric_specs = []
    written_files = []
    parts = []

    for model in model_names:
        src = output_dir / dataset_name / model / f"silhouette_stats_recomputed.{model}.{dataset_name}.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if not {"projection", "model", "mean_sil", "std_sil"}.issubset(df.columns):
            continue
        tmp = df.copy()
        tmp["dataset"] = tmp["projection"].astype(str).map(lambda p: f"{dataset_name}__{p}")
        tmp["metric_mean"] = tmp["mean_sil"]
        tmp["metric_std"] = tmp["std_sil"].fillna(0.0)
        tmp["n_runs"] = np.nan
        parts.append(tmp[["dataset", "model", "metric_mean", "metric_std", "n_runs"]])

    if not parts:
        return metric_specs, written_files

    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.rename(columns={"metric_mean": "mean_sil", "metric_std": "std_sil"})
    out_path = demsar_input_dir / "silhouette_mean.csv"
    out_df.to_csv(out_path, index=False)
    written_files.append(out_path)

    metric_specs.append({
        "root_dir": str(output_dir),
        "output_dir": str(demsar_input_dir.parent / "demsar_results" / "silhouette_mean"),
        "alpha": alpha,
        "input_csv": str(out_path),
        "dataset_col": "dataset",
        "model_col": "model",
        "score_col": "mean_sil",
        "metric_name": "silhouette_mean",
        "higher_is_better": True,
    })
    return metric_specs, written_files


def build_intrinsic_id_inputs(
    output_dir: Path,
    dataset_name: str,
    model_names: list[str],
    demsar_input_dir: Path,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[Path]]:
    metric_specs = []
    written_files = []
    parts = []

    for model in model_names:
        src = output_dir / dataset_name / model / f"intrinsic_dimension_summary.{model}.{dataset_name}.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if not {"model", "run", "method", "k", "global_id"}.issubset(df.columns):
            continue
        tmp = aggregate_mean_std(
            df=df,
            group_cols=["model", "method", "k"],
            value_col="global_id",
            mean_name="mean_global_id",
            std_name="std_global_id",
        )
        tmp["dataset"] = tmp.apply(lambda r: f"{dataset_name}__{r['method']}__k{r['k']}", axis=1)
        parts.append(tmp[["dataset", "model", "mean_global_id", "std_global_id", "n_runs"]])

    if not parts:
        return metric_specs, written_files

    out_df = pd.concat(parts, ignore_index=True)
    out_path = demsar_input_dir / "global_id.csv"
    out_df.to_csv(out_path, index=False)
    written_files.append(out_path)

    metric_specs.append({
        "root_dir": str(output_dir),
        "output_dir": str(demsar_input_dir.parent / "demsar_results" / "global_id"),
        "alpha": alpha,
        "input_csv": str(out_path),
        "dataset_col": "dataset",
        "model_col": "model",
        "score_col": "mean_global_id",
        "metric_name": "global_id",
        "higher_is_better": False,
    })
    return metric_specs, written_files


def _build_pairwise_meanstd_frame(
    df: pd.DataFrame,
    dataset_name: str,
    value_col: str,
    mean_col: str,
    std_col: str,
) -> pd.DataFrame:
    grouped = aggregate_mean_std(
        df=df,
        group_cols=["projection", "model_a", "model_b"],
        value_col=value_col,
        mean_name=mean_col,
        std_name=std_col,
    )
    grouped["dataset"] = grouped.apply(
        lambda r: f"{dataset_name}__{r['projection']}__vs__{r['model_b']}",
        axis=1,
    )
    grouped = grouped.rename(columns={"model_a": "model"})
    return grouped[["dataset", "model", mean_col, std_col, "n_runs"]]


def build_pairwise_kernel_inputs(
    output_dir: Path,
    dataset_name: str,
    demsar_input_dir: Path,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[Path]]:
    metric_specs = []
    written_files = []

    ece_src = output_dir / dataset_name / "pairwise_ece_summary.csv"
    if ece_src.exists():
        df = pd.read_csv(ece_src)
        if {"run", "projection", "model_a", "model_b", "ece"}.issubset(df.columns):
            out_df = _build_pairwise_meanstd_frame(
                df=df,
                dataset_name=dataset_name,
                value_col="ece",
                mean_col="mean_ece",
                std_col="std_ece",
            )
            out_path = demsar_input_dir / "pairwise_ece.csv"
            out_df.to_csv(out_path, index=False)
            written_files.append(out_path)

            metric_specs.append({
                "root_dir": str(output_dir),
                "output_dir": str(demsar_input_dir.parent / "demsar_results" / "pairwise_ece"),
                "alpha": alpha,
                "input_csv": str(out_path),
                "dataset_col": "dataset",
                "model_col": "model",
                "score_col": "mean_ece",
                "metric_name": "pairwise_ece",
                "higher_is_better": False,
            })

    dist_src = output_dir / dataset_name / "pairwise_distance_summary.csv"
    if dist_src.exists():
        df = pd.read_csv(dist_src)
        if {"run", "projection", "model_a", "model_b", "distance"}.issubset(df.columns):
            out_df = _build_pairwise_meanstd_frame(
                df=df,
                dataset_name=dataset_name,
                value_col="distance",
                mean_col="mean_distance",
                std_col="std_distance",
            )
            out_path = demsar_input_dir / "pairwise_distance.csv"
            out_df.to_csv(out_path, index=False)
            written_files.append(out_path)

            metric_specs.append({
                "root_dir": str(output_dir),
                "output_dir": str(demsar_input_dir.parent / "demsar_results" / "pairwise_distance"),
                "alpha": alpha,
                "input_csv": str(out_path),
                "dataset_col": "dataset",
                "model_col": "model",
                "score_col": "mean_distance",
                "metric_name": "pairwise_distance",
                "higher_is_better": False,
            })

    return metric_specs, written_files


def build_all_demsar_inputs(
    output_dir: Path,
    dataset_name: str,
    model_names: list[str],
    alpha: float,
) -> tuple[list[dict[str, Any]], list[Path]]:
    demsar_input_dir = output_dir / "demsar_inputs"
    demsar_input_dir.mkdir(parents=True, exist_ok=True)

    metric_specs: list[dict[str, Any]] = []
    written_files: list[Path] = []

    specs, files = build_silhouette_inputs(output_dir, dataset_name, model_names, demsar_input_dir, alpha)
    metric_specs.extend(specs)
    written_files.extend(files)

    specs, files = build_intrinsic_id_inputs(output_dir, dataset_name, model_names, demsar_input_dir, alpha)
    metric_specs.extend(specs)
    written_files.extend(files)

    specs, files = build_pairwise_kernel_inputs(output_dir, dataset_name, demsar_input_dir, alpha)
    metric_specs.extend(specs)
    written_files.extend(files)

    return metric_specs, written_files


def build_main_metric_summary(demsar_input_files: list[Path], output_csv: Path) -> None:
    parts = []
    for path in demsar_input_files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        parts.append(df)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(output_csv, index=False)


def load_yaml_config(path: Path) -> dict[str, Any]:
    cfg = read_yaml(str(path))
    if cfg is None:
        raise ValueError(f"YAML config is empty or unreadable: {path}")
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML config must be a dictionary")
    return cfg


def demsar_wrapper(cfg):
    out_dir = Path(cfg["out_dir"])
    dataset_name = resolve_dataset_name(cfg)
    alpha = float(cfg.get("alpha", 0.05))
    model_names = resolve_model_names(cfg)

    metric_specs, written_files = build_all_demsar_inputs(
        output_dir=out_dir,
        dataset_name=dataset_name,
        model_names=model_names,
        alpha=alpha,
    )

    build_main_metric_summary(
        demsar_input_files=written_files,
        output_csv=out_dir / "demsar_inputs" / "all_metrics_mean_std_summary.csv",
    )

    for metric_cfg in metric_specs:
        run_demsar(metric_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    demsar_wrapper(cfg)


