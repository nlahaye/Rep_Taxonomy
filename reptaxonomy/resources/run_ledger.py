"""Resource ledger: energy, carbon, water and compute of any command, plus per-arm summaries.

Wraps one command (training, inference, export ...) and writes one JSON record:
wall / CPU time, peak RSS of the command's processes, peak GPU memory of the GPUs it was
given, energy (CodeCarbon: GPU via NVML, CPU via RAPL or TDP model, RAM modelled), CO2e and
water. Optionally merges a FLOPs / parameter-count JSON written by the model side.

  python -m reptaxonomy.resources.run_ledger run --arm conv_dbn --stage train --out LEDGER_DIR \
      [--gpus 0] [--carbon-intensity 220] -- python -m sit_fuse... -y ...
  python -m reptaxonomy.resources.run_ledger summary --ledger LEDGER_DIR --out summary.csv \
      [--skill-csv skill.csv --skill-arm-col arm --skill-col value] [--flops-dir DIR]

Carbon: CodeCarbon offline tracker (country / region fixed, no network), and, when
--carbon-intensity (g CO2e / kWh) is given, a pinned recomputation kg = kWh * ci / 1000 so
arms run months apart use the same grid factor.
Water (L) = kWh * (WUE_site + PUE * EWIF): WUE_site = on-site cooling water per kWh (0 for an
air-cooled workstation), EWIF = water consumed per kWh of generated electricity (default
3.14 L/kWh, US average; low / high 0.9 / 5.0 L/kWh reported as a range).
Measurements are only attributable when the command has the machine (or the GPU) to itself:
CodeCarbon's GPU reading is the whole device; CPU / RAM power is machine level.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import threading
import time
from typing import Any

EWIF_DEFAULT = 3.14
EWIF_RANGE = (0.9, 5.0)


def _gpu_mem_poll(gpus: list[int], stop: threading.Event, peak: dict, every: float = 2.0) -> None:
    q = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    while not stop.is_set():
        try:
            out = subprocess.run(q, capture_output=True, text=True, timeout=10).stdout
            for line in out.strip().splitlines():
                i, m = [s.strip() for s in line.split(",")[:2]]
                if not gpus or int(i) in gpus:
                    peak[int(i)] = max(peak.get(int(i), 0.0), float(m))
        except Exception:
            pass
        stop.wait(every)


def _tracker(out_dir: str, name: str, country: str, region: str | None, gpus: list[int], interval: float):
    try:
        from codecarbon import OfflineEmissionsTracker
    except Exception:
        return None
    kw: dict[str, Any] = dict(output_dir=out_dir, output_file=f"{name}.emissions.csv", project_name=name,
                              country_iso_code=country, measure_power_secs=interval, log_level="error",
                              save_to_file=True, tracking_mode="machine")
    if region:
        kw["region"] = region
    if gpus:
        kw["gpu_ids"] = gpus
    try:
        return OfflineEmissionsTracker(**kw)
    except TypeError:
        kw.pop("gpu_ids", None)
        return OfflineEmissionsTracker(**kw)


def _tracker_fields(tracker) -> dict[str, Any]:
    d = getattr(tracker, "final_emissions_data", None)
    if d is None:
        return {}
    keep = ("energy_consumed", "cpu_energy", "gpu_energy", "ram_energy", "emissions", "emissions_rate",
            "cpu_power", "gpu_power", "ram_power", "country_iso_code", "region", "cpu_model", "gpu_model",
            "codecarbon_version", "tracking_mode", "duration")
    return {k: getattr(d, k, None) for k in keep}


def water_litres(kwh: float, wue_site: float, pue: float, ewif: float) -> float:
    return float(kwh) * (float(wue_site) + float(pue) * float(ewif))


def cmd_run(a) -> int:
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("no command given (put it after --)")
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.arm}.{a.stage}" + (f".{a.tag}" if a.tag else "")
    rec_path = os.path.join(a.out, name + ".json")
    if os.path.exists(rec_path) and not a.overwrite:
        prev = json.load(open(rec_path))
        if prev.get("returncode") == 0:
            print(f"{rec_path} exists (rc 0); skipping (use --overwrite to remeasure)")
            return 0
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else []
    tracker = _tracker(a.out, name, a.country, a.region, gpus, a.interval)
    if tracker is None:
        print("run_ledger: codecarbon not importable; energy, carbon and water will be 0", file=sys.stderr)
    stop, peak = threading.Event(), {}
    poll = threading.Thread(target=_gpu_mem_poll, args=(gpus, stop, peak), daemon=True)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    if tracker is not None:
        try:
            tracker.start()
        except Exception as e:           # noqa: BLE001  measurement must not stop the job
            print(f"run_ledger: codecarbon start failed ({e})", file=sys.stderr)
            tracker = None
    poll.start()
    try:
        rc = subprocess.call(cmd)
    finally:
        stop.set()
        poll.join(timeout=15)
        if tracker is not None:
            try:
                tracker.stop()
            except Exception as e:       # noqa: BLE001
                print(f"run_ledger: codecarbon stop failed ({e})", file=sys.stderr)
    wall = time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cc = _tracker_fields(tracker) if tracker is not None else {}
    kwh = float(cc.get("energy_consumed") or 0.0)
    rec: dict[str, Any] = {
        "arm": a.arm, "stage": a.stage, "tag": a.tag, "cmd": cmd, "returncode": rc,
        "start_unix": t0, "wall_s": wall,
        "cpu_user_s": r1.ru_utime - r0.ru_utime, "cpu_sys_s": r1.ru_stime - r0.ru_stime,
        "peak_rss_gb": r1.ru_maxrss / 1024 ** 2,         # largest single child process (Linux: KiB)
        "peak_gpu_mem_gb": {str(k): v / 1024 for k, v in peak.items()},
        "energy_kwh": kwh, "codecarbon": cc,
        "co2e_kg_codecarbon": cc.get("emissions"),
        "carbon_intensity_g_per_kwh": a.carbon_intensity,
        "co2e_kg_pinned": (kwh * a.carbon_intensity / 1000.0) if a.carbon_intensity else None,
        "water": {"wue_site_l_per_kwh": a.wue_site, "pue": a.pue, "ewif_l_per_kwh": a.ewif,
                  "litres": water_litres(kwh, a.wue_site, a.pue, a.ewif),
                  "litres_low": water_litres(kwh, a.wue_site, a.pue, EWIF_RANGE[0]),
                  "litres_high": water_litres(kwh, a.wue_site, a.pue, EWIF_RANGE[1])},
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "gpus": gpus, "exclusive": bool(a.exclusive), "measured": tracker is not None,
    }
    if a.flops_json and os.path.exists(a.flops_json):
        rec["flops"] = json.load(open(a.flops_json))
    tmp = rec_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    os.replace(tmp, rec_path)
    print(json.dumps({k: rec[k] for k in ("arm", "stage", "returncode", "wall_s", "energy_kwh")}))
    return rc


def cmd_summary(a) -> None:
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(os.path.join(a.ledger, "*.json"))):
        r = json.load(open(p))
        if "arm" not in r:
            continue
        rows.append({"arm": r["arm"], "stage": r["stage"], "tag": r.get("tag"), "returncode": r.get("returncode"),
                     "wall_h": r["wall_s"] / 3600, "cpu_h": (r["cpu_user_s"] + r["cpu_sys_s"]) / 3600,
                     "peak_rss_gb": r["peak_rss_gb"],
                     "peak_gpu_mem_gb": max(r.get("peak_gpu_mem_gb", {}).values() or [0.0]),
                     "energy_kwh": r["energy_kwh"],
                     "gpu_kwh": (r.get("codecarbon") or {}).get("gpu_energy"),
                     "cpu_kwh": (r.get("codecarbon") or {}).get("cpu_energy"),
                     "ram_kwh": (r.get("codecarbon") or {}).get("ram_energy"),
                     "co2e_kg": r.get("co2e_kg_pinned") if r.get("co2e_kg_pinned") is not None
                     else r.get("co2e_kg_codecarbon"),
                     "water_l": r["water"]["litres"], "water_l_low": r["water"]["litres_low"],
                     "water_l_high": r["water"]["litres_high"], "exclusive": r.get("exclusive")})
    if not rows:
        raise SystemExit(f"no ledger records in {a.ledger}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.splitext(a.out)[0] + ".runs.csv", index=False)
    num = ["wall_h", "cpu_h", "energy_kwh", "gpu_kwh", "cpu_kwh", "ram_kwh", "co2e_kg", "water_l", "water_l_low",
           "water_l_high"]
    g = df[df["returncode"] == 0].groupby(["arm", "stage"])
    out = g[num].sum(min_count=1)
    out[["peak_rss_gb", "peak_gpu_mem_gb"]] = g[["peak_rss_gb", "peak_gpu_mem_gb"]].max()
    out["n_runs"] = g.size()
    out = out.reset_index()
    tot = out.groupby("arm")[num].sum(min_count=1).reset_index()
    tot["stage"] = "total"
    out = pd.concat([out, tot], ignore_index=True)
    if a.flops_dir:
        fl = []
        for p in glob.glob(os.path.join(a.flops_dir, "*.flops.json")):
            r = json.load(open(p))
            fl.append({"arm": r.get("arm"), **{k: r.get(k) for k in ("params_encoder", "params_total",
                                                                       "flops_per_pixel", "macs_per_pixel")}})
        if fl:
            out = out.merge(pd.DataFrame(fl), on="arm", how="left")
    if a.skill_csv:
        files = [f for pat in a.skill_csv for f in sorted(glob.glob(pat))]
        if not files:
            raise SystemExit(f"no skill CSV matches {a.skill_csv}")
        sk = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)   # mean over files (e.g. targets)
        if a.skill_filter:
            for kv in a.skill_filter:
                k, v = kv.split("=", 1)
                sk = sk[sk[k].astype(str) == v]
        sk = sk.groupby(a.skill_arm_col)[a.skill_col].mean().rename("skill").reset_index()
        sk = sk.rename(columns={a.skill_arm_col: "arm"})
        if a.skill_arm_suffix:
            sk["arm"] = sk["arm"].str.replace(a.skill_arm_suffix + "$", "", regex=True)
        out = out.merge(sk, on="arm", how="left")
        tot_e = out[out["stage"] == "total"].set_index("arm")["energy_kwh"]
        out["skill_per_kwh_total"] = out["skill"] / out["arm"].map(tot_e.where(tot_e > 0))
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--stage", required=True, help="train | inference | export | ...")
    r.add_argument("--tag", default="", help="extra key, e.g. a window or scene set")
    r.add_argument("--out", required=True, help="ledger directory")
    r.add_argument("--gpus", default="", help="GPU ids given to the command, e.g. 0")
    r.add_argument("--country", default="USA")
    r.add_argument("--region", default="california")
    r.add_argument("--interval", type=float, default=15.0, help="CodeCarbon sampling interval, s")
    r.add_argument("--carbon-intensity", type=float, default=None, help="pinned g CO2e / kWh")
    r.add_argument("--wue-site", type=float, default=0.0, help="on-site cooling water, L / kWh")
    r.add_argument("--pue", type=float, default=1.0)
    r.add_argument("--ewif", type=float, default=EWIF_DEFAULT, help="L / kWh of generated electricity")
    r.add_argument("--flops-json", default=None)
    r.add_argument("--exclusive", action="store_true", help="the command had the machine / GPU to itself")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    s = sub.add_parser("summary")
    s.add_argument("--ledger", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--flops-dir", default=None)
    s.add_argument("--skill-csv", action="append", help="skill table(s), glob allowed (repeatable)")
    s.add_argument("--skill-arm-col", default="variant")
    s.add_argument("--skill-col", default="value")
    s.add_argument("--skill-arm-suffix", default="", help="strip from skill arm names, e.g. _env")
    s.add_argument("--skill-filter", action="append", help="col=value rows to keep (repeatable)")
    a = p.parse_args(argv)
    if a.mode == "run":
        return cmd_run(a)
    cmd_summary(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
