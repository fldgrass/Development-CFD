#!/usr/bin/env python3
"""자체 테스트: 배치 해석 (유속 1,2 m/s × 영각 45,90° = 4케이스, 최소 옵션).
새 격자 축소·배치 프리셋 반영·Cd 추출을 검증한다. UI 없이 직접 실행."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "modules"))
from cfd_manager import BatchAnalysisManager, RESULTS_DIR

STL = Path(__file__).parent / "stl_uploads" / "onemesh3.stl"
OUT_CSV = RESULTS_DIR / "unit_cell" / "selftest_batch.csv"
if OUT_CSV.exists():
    OUT_CSV.unlink()

def log(msg):
    msg = msg.strip()
    if not msg:
        return
    import re
    if re.search(r"Time = [0-9]", msg) or any(
        x in msg for x in ["✅", "❌", "▶", "완료", "FATAL", "Error",
                            "cells:", "Finished meshing", "케이스", "오류"]):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

print("=" * 60, flush=True)
print("  자체 테스트: 배치 (U=1,2 × α=45,90, 최소 옵션)", flush=True)
print("  최소: end_time=500, refine=2, residual=1e-3", flush=True)
print("=" * 60, flush=True)

mgr = BatchAnalysisManager(
    mode="unit_cell",
    stl_paths={"net": STL},
    speeds=[1.0, 2.0],
    angles=[45.0, 90.0],
    output_csv=OUT_CSV,
    common_params={
        "n_cores": 16, "nx": 1, "ny": 1,
        "end_time": 500, "refine_level": 2,
        "residual_control": 1e-3, "write_interval": 100,
    },
    progress_cb=lambda p, s, e, label="": print(
        f"  [진행 {p:.1f}%] 케이스 {s}/{e}  {label}", flush=True
    ) if int(p) % 10 == 0 else None,
    log_cb=log,
)
t0 = time.time()
mgr.run_batch()
print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)
print(f"결과 CSV: {OUT_CSV}", flush=True)
