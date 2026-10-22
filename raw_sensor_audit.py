"""Read-only audit of the continuous FarmGuard MQTT sensor log."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

EXPECTED = ["timestamp", "topic", "pir", "ldr", "beam_blocked", "gas", "temperature", "state"]
SENSORS = ["pir", "ldr", "beam_blocked", "gas", "temperature"]

def find_sensor_csv(data_dir="data") -> Path:
    candidates=[]
    for path in Path(data_dir).glob("*.csv"):
        try:
            cols=set(pd.read_csv(path,nrows=2).columns)
            if set(EXPECTED).issubset(cols): candidates.append(path)
        except Exception: pass
    if not candidates: raise FileNotFoundError("No continuous FarmGuard sensor CSV found")
    return max(candidates,key=lambda p:p.stat().st_size)

def audit_raw_sensor_csv(path=None, output="results/raw_sensor_audit.txt"):
    path=Path(path) if path else find_sensor_csv(); df=pd.read_csv(path); lines=[]
    lines += ["FARMGUARD-AI RAW SENSOR LOG AUDIT", "="*38, f"Source: {path.resolve()}", "Original file modified: NO", f"Rows: {len(df)}", f"Columns: {', '.join(df.columns)}", ""]
    lines.append("MISSING VALUES"); lines += [f"  {c}: {int(v)}" for c,v in df.isna().sum().items()]; lines += ["",f"Exact duplicate rows: {int(df.duplicated().sum())}",""]
    lines.append("NUMERIC VALIDITY AND RANGES")
    numeric={}
    for c in SENSORS:
        values=pd.to_numeric(df[c],errors="coerce"); numeric[c]=values
        lines.append(f"  {c}: invalid={int(values.isna().sum()-df[c].isna().sum())}, min={values.min()}, max={values.max()}, mean={values.mean():.3f}")
    lines += ["", "SENSOR COUNTS AND DISTRIBUTIONS"]
    lines += [f"  PIR 0: {int((numeric['pir']==0).sum())}", f"  PIR 1: {int((numeric['pir']==1).sum())}", f"  Beam clear (0): {int((numeric['beam_blocked']==0).sum())}", f"  Beam blocked (1): {int((numeric['beam_blocked']==1).sum())}"]
    for c in ("ldr", "gas", "temperature"):
        q=numeric[c].quantile([0,.05,.25,.5,.75,.95,1]); lines.append(f"  {c} percentiles: min={q.loc[0]:.3f}, p05={q.loc[.05]:.3f}, p25={q.loc[.25]:.3f}, median={q.loc[.5]:.3f}, p75={q.loc[.75]:.3f}, p95={q.loc[.95]:.3f}, max={q.loc[1]:.3f}")
    lines += ["", "SENSOR NOISE / STEP CHANGES"]
    for c in ("ldr", "gas", "temperature"):
        steps=numeric[c].diff().abs().dropna(); lines.append(f"  {c}: median absolute step={steps.median():.3f}, p95 absolute step={steps.quantile(.95):.3f}, max absolute step={steps.max():.3f}")
    lines += ["", "STATE COUNTS"]
    lines += [f"  {k}: {v}" for k,v in df["state"].value_counts(dropna=False).items()]
    parsed=pd.to_datetime(df["timestamp"],errors="coerce",utc=True); intervals=parsed.sort_values().diff().dt.total_seconds().dropna()
    lines += ["", "MQTT SAMPLING INTERVALS", f"  Invalid timestamps: {int(parsed.isna().sum())}"]
    if len(intervals):
        positive=intervals[intervals>0]
        lines += [f"  Zero-second intervals: {int((intervals==0).sum())}", f"  Negative intervals: {int((intervals<0).sum())}", f"  Median positive interval: {positive.median() if len(positive) else 'unavailable'} sec", f"  Maximum interval: {intervals.max()} sec"]
    lines += ["", "SOFTWARE / LOGGING CHECKS"]
    checks={
        "PIR outside 0/1": int((~numeric["pir"].isin([0,1]) & numeric["pir"].notna()).sum()),
        "beam_blocked outside 0/1": int((~numeric["beam_blocked"].isin([0,1]) & numeric["beam_blocked"].notna()).sum()),
        "negative LDR": int((numeric["ldr"]<0).sum()), "negative gas": int((numeric["gas"]<0).sum()),
        "unknown state": int((~df["state"].isin(["NORMAL","WATCH","CONFIRMED"])).sum()),
        "LDR outside ESP32 ADC range 0..4095": int(((numeric["ldr"]<0)|(numeric["ldr"]>4095)).sum()),
        "gas outside ESP32 ADC range 0..4095": int(((numeric["gas"]<0)|(numeric["gas"]>4095)).sum()),
        "temperature outside plausible DHT range -40..80 C": int(((numeric["temperature"]<-40)|(numeric["temperature"]>80)).sum()),
        "temperature equal to 0 C (likely startup/default reading)": int((numeric["temperature"]==0).sum()),
    }
    lines += [f"  {k}: {v}" for k,v in checks.items()]
    corrupted=sum(checks.values())
    lines += ["", "AUDIT VERDICT", f"  Obvious corrupted sensor/state readings: {corrupted}", "  PASS: no obvious corruption detected." if corrupted==0 else "  REVIEW REQUIRED: one or more obvious range/state violations were detected.", "", "NOTES", "  Exact duplicates are reported but the original file is not changed.", "  Long timestamp gaps commonly represent dashboard/device downtime rather than sensor corruption.", "  Timestamps have whole-second precision, so sub-second MQTT intervals cannot be reconstructed reliably.", "  The rule-generated state is not independent ground truth and must not be used as an ML target.", "  This log is suitable for ranges, calibration, plots, and debugging only."]
    report="\n".join(lines)+"\n"; out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(report,encoding="utf-8"); return report

if __name__=="__main__": print(audit_raw_sensor_csv())
