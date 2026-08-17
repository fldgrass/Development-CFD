"""
cfd_manager.py
==============
OpenFOAM CFD 해석 관리 백엔드 모듈
- 해석 케이스 생성/실행/모니터링
- 주기 경계조건 자동 설정
- MPI 병렬화 자동 구성
- 결과 CSV 자동 추출
"""

import os
import re
import shutil
import subprocess
import threading
import math
import json
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Callable, Any

# ─── 로깅 설정 ─────────────────────────────────────────────────────────────
logger = logging.getLogger("cfd_manager")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════
# 경로 상수
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR       = Path(__file__).resolve().parent.parent
OF_TEMPLATES   = BASE_DIR / "openfoam"
RESULTS_DIR    = BASE_DIR / "results"
STL_UPLOAD_DIR = BASE_DIR / "stl_uploads"
LOGS_DIR       = BASE_DIR / "logs"

for _d in [RESULTS_DIR / "unit_cell", RESULTS_DIR / "full_structure",
           STL_UPLOAD_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════════════════

def get_cpu_count() -> int:
    """물리 CPU 코어 수 반환 (하이퍼스레딩 제외).
    Open MPI는 기본적으로 물리 코어 수만큼만 슬롯을 허용하므로,
    논리 코어(HT) 수로 mpirun을 호출하면 'not enough slots' 오류가 발생한다.
    """
    try:
        # /proc/cpuinfo 에서 물리 소켓 × 코어/소켓 계산
        with open("/proc/cpuinfo") as f:
            content = f.read()
        physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", content))
        cores = re.findall(r"cpu cores\s*:\s*(\d+)", content)
        if physical_ids and cores:
            n = len(physical_ids) * int(cores[0])
            return min(max(n, 4), 32)
    except Exception:
        pass
    try:
        # fallback: os.cpu_count() 의 절반 (HT 감안)
        n = (os.cpu_count() or 8) // 2
        return min(max(n, 4), 32)
    except Exception:
        return 8


def compute_velocity_vector(speed: float, angle_deg: float) -> Tuple[float, float, float]:
    """
    유속 크기와 영각(AoA)으로 속도 벡터 계산
    Args:
        speed: 유속 [m/s]
        angle_deg: 영각 [degree]
    Returns:
        (Ux, Uy, Uz) 속도 벡터
    """
    angle_rad = math.radians(angle_deg)
    ux = speed * math.cos(angle_rad)
    uz = speed * math.sin(angle_rad)
    return (ux, 0.0, uz)


def coordinate_convention(angle_deg: float, mode: str) -> Dict:
    """좌표/영각 단일 규약 — Unit Cell 을 기준 구현으로 삼아 두 모드가 '유속과
    그물면 법선 사이의 상대각(= 90°−α)'을 동일하게 갖도록 정의한다.

    - unit_cell : 그물면 고정(법선 z), 유속을 회전 (cosα,0,sinα). 주기(cyclic) BC 라
      어떤 각도든 정상 작동. dragDir=유속방향, liftDir=유속에 수직(XZ 평면).
    - full_structure : 풍동(wind-tunnel) 방식 — 유속을 x 로 고정하고 그물 형상을
      Y 축으로 α 회전(법선 z→x). 같은 상대각을 만들면서, 고정 x-inlet 에 항상 정상
      유입(법선속도 = U)되어 고각에서 힘이 붕괴하지 않는다. dragDir=x, liftDir=z.
      (두 모드는 Y 축 α 회전으로 서로 연결되는 동일 물리계 → Cd/Cl 일치.)

    반환: inlet(유속 방향 단위벡터), normal(그물면 법선), drag, lift, pitch,
    relative_angle_deg(유속-법선 상대각), rotate_geometry_deg(형상 회전각, Y축).
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    if mode == "unit_cell":
        inlet  = (ca, 0.0, sa)        # 유속 방향(회전)
        normal = (0.0, 0.0, 1.0)      # 그물면 법선(고정)
        drag   = (ca, 0.0, sa)
        lift   = (-sa, 0.0, ca)
        rot    = 0.0                  # 형상 회전 없음
    else:  # full_structure (풍동식)
        inlet  = (1.0, 0.0, 0.0)      # 유속 고정(x)
        normal = (sa, 0.0, ca)        # 그물면 법선 = Y축 α 회전(z→x)
        drag   = (1.0, 0.0, 0.0)
        lift   = (0.0, 0.0, 1.0)
        rot    = float(angle_deg)     # 형상을 Y축으로 α 회전
    pitch = (0.0, 1.0, 0.0)           # pitchAxis = y (양 모드 공통)
    _dot = max(-1.0, min(1.0, sum(inlet[i]*normal[i] for i in range(3))))
    rel = math.degrees(math.acos(_dot))
    return {"inlet": inlet, "normal": normal, "drag": drag, "lift": lift,
            "pitch": pitch, "relative_angle_deg": rel,
            "rotate_geometry_deg": rot, "mode": mode}


# ───────────────────────────────────────────────────────────────────────────
# 통일 표시 좌표계 (Display / User frame) — 두 모드 공통의 단일 규약
# ───────────────────────────────────────────────────────────────────────────
#   전역 축:  +X = 그물면 법선(정면), X–Y 평면 = 수면, +Z = 수심(깊어지는 방향)
#   그물 형상은 Y–Z 평면에 놓이고(법선 +X), 유속은 X–Y(수면) 평면에서 회전한다.
#   영각(AoA) 정의 (두 모드 동일):
#     AoA = 90° → 유속 −X→+X (그물 정면 충돌, 법선 입사)
#     AoA = 0°  → 유속 +Y→−Y (그물면 평행, 그레이징)
#   유속 단위벡터  d(AoA) = (sin AoA, −cos AoA, 0)
#
#   ※ 솔버 내부 좌표(coordinate_convention)는 수치 안정성을 위해 모드별로
#     다르게 두되(검증 완료), 화면·리포트·시각화는 이 표시 좌표계로 통일한다.
#     솔버→표시 변환은 solver_to_display_rotation() 이 담당한다.
def display_convention(angle_deg: float) -> Dict:
    """두 모드 공통의 통일 표시 좌표계 벡터를 반환한다(모드 무관).

    반환: flow/inlet(유속), normal(그물면 법선=+X), drag(=유속), lift(유속 수직,
    수면 내), depth/pitch(=+Z 수심), relative_angle_deg(유속–법선 상대각).
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    flow   = (sa, -ca, 0.0)      # d(AoA): 90°→+X, 0°→−Y
    normal = (1.0, 0.0, 0.0)     # 그물면 법선 (Y–Z 평면)
    lift   = (ca, sa, 0.0)       # 양력: 유속에 수직, 수면(X–Y) 평면 내
    depth  = (0.0, 0.0, 1.0)     # 수심 = pitch 축
    _dot = max(-1.0, min(1.0, flow[0]*normal[0] + flow[1]*normal[1] + flow[2]*normal[2]))
    rel = math.degrees(math.acos(_dot))
    return {"flow": flow, "inlet": flow, "normal": normal, "drag": flow,
            "lift": lift, "depth": depth, "pitch": depth,
            "relative_angle_deg": rel, "frame": "display"}


def solver_to_display_rotation(mode: str, angle_deg: float) -> List[List[float]]:
    """모드별 솔버 좌표 → 통일 표시 좌표 회전행렬 R(3×3, 행 우선).

    v_display = R · v_solver. 시각화에서 형상·벡터장을 통일 프레임으로 표시하거나
    힘 벡터를 통일 프레임으로 보고할 때 사용한다. 두 R 모두 det=+1 정상회전.

    - unit_cell (각도 무관): 솔버(법선 +Z, 유속 (cosα,0,sinα))
        → 표시(법선 +X, 유속 (sinα,−cosα,0)).
    - full_structure (각도 의존): 솔버(유속 +X 고정, 그물 Y축 α 회전)
        → 표시(법선 +X, 유속 (sinα,−cosα,0)).
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    if mode == "unit_cell":
        return [[0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0]]
    # full_structure
    return [[sa, 0.0, ca],
            [-ca, 0.0, sa],
            [0.0, -1.0, 0.0]]


def apply_rotation(R: List[List[float]], v) -> Tuple[float, float, float]:
    """회전행렬 R(3×3)을 벡터 v(3)에 적용: R·v."""
    return (R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
            R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
            R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2])


def solver_flow_vector(mode: str, angle_deg: float) -> Tuple[float, float, float]:
    """모드별 솔버 좌표계에서의 유속 단위벡터(시각화·검증용)."""
    a = math.radians(angle_deg)
    if mode == "unit_cell":
        return (math.cos(a), 0.0, math.sin(a))
    return (1.0, 0.0, 0.0)   # full_structure: 풍동식(유속 +X 고정)


def verify_coordinate_consistency(angles=(0, 30, 45, 60, 90), tol_deg=1.0) -> Dict:
    """항목2·3: 동일 입력에 대해 두 모드의 좌표 규약을 비교하는 자동 일관성 검증.
    유속-그물면 상대각이 일치(≤tol_deg)하는지, dragDir 가 유속 방향과 정렬되는지,
    pitchAxis 가 일치하는지 확인하고, 허용오차 초과 시 경고를 생성한다.
    반환: {angle: {uc, fs, rel_diff_deg, drag_aligned_uc, drag_aligned_fs, ok, warnings}}.
    """
    report = {}
    for a in angles:
        disp = display_convention(a)
        res = {}
        max_err = 0.0
        for m in ("unit_cell", "full_structure"):
            R = solver_to_display_rotation(m, a)
            sflow = solver_flow_vector(m, a)                 # 솔버 유속
            snorm = coordinate_convention(a, m)["normal"]    # 솔버 그물면 법선
            dflow = apply_rotation(R, sflow)                 # → 표시 프레임 유속
            dnorm = apply_rotation(R, snorm)                 # → 표시 프레임 법선
            ferr = max(abs(dflow[i] - disp["flow"][i]) for i in range(3))
            nerr = max(abs(dnorm[i] - disp["normal"][i]) for i in range(3))
            _d = max(-1.0, min(1.0, sum(dflow[i]*dnorm[i] for i in range(3))))
            res[m] = {"flow_display": tuple(round(x, 4) for x in dflow),
                      "normal_display": tuple(round(x, 4) for x in dnorm),
                      "relative_angle_deg": math.degrees(math.acos(_d)),
                      "flow_err": ferr, "normal_err": nerr}
            max_err = max(max_err, ferr, nerr)
        rel_diff = abs(res["unit_cell"]["relative_angle_deg"]
                       - res["full_structure"]["relative_angle_deg"])
        warns = []
        if max_err > 1e-3:
            warns.append(f"솔버→표시 변환 후 유속/법선 오차 {max_err:.1e} > 1e-3")
        if rel_diff > tol_deg:
            warns.append(f"상대각 차이 {rel_diff:.2f}° > 허용 {tol_deg}°")
        report[a] = {"uc": res["unit_cell"], "fs": res["full_structure"],
                     "rel_diff_deg": rel_diff,
                     "drag_aligned_uc": 1.0 - res["unit_cell"]["flow_err"],
                     "drag_aligned_fs": 1.0 - res["full_structure"]["flow_err"],
                     "display_flow": tuple(round(x, 4) for x in disp["flow"]),
                     "ok": not warns, "warnings": warns}
    return report


def rotate_points_y(pts, angle_deg, center=(0.0, 0.0, 0.0)):
    """점 목록을 center 기준 Y축으로 angle_deg 회전. (x,y,z) →
    ((x-cx)cosθ+(z-cz)sinθ+cx, y, -(x-cx)sinθ+(z-cz)cosθ+cz)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cy, cz = center
    out = []
    for (x, y, z) in pts:
        dx, dz = x - cx, z - cz
        out.append((dx*ca + dz*sa + cx, y, -dx*sa + dz*ca + cz))
    return out


def compute_turbulence_params(speed: float, intensity: float = 0.05,
                               length_scale: float = 0.02) -> Dict[str, float]:
    """
    난류 초기 조건 계산 (k-omega SST 기준)
    Args:
        speed: 유속 [m/s]
        intensity: 난류 강도 (기본 5%)
        length_scale: 혼합 길이 [m]
    Returns:
        {'k': ..., 'omega': ...}
    """
    k = 1.5 * (intensity * speed) ** 2
    cmu = 0.09
    omega = math.sqrt(k) / (cmu ** 0.25 * length_scale)
    return {"k": k, "omega": omega}


def replace_in_file(filepath: Path, replacements: Dict[str, str]) -> None:
    """파일 내 텍스트 치환 (OpenFOAM 딕셔너리 자동 수정)"""
    text = filepath.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    filepath.write_text(text, encoding="utf-8")


def steady_force_convergence(case_dir: Path,
                             window: float = 0.2) -> Dict[str, Any]:
    """정상 해석의 힘 계수가 실제로 수렴했는지 판정한다(요구서 §17).

    잔차만으로 종료를 판단하면 '잔차는 내려갔는데 Cd 는 아직 흐르는' 상태를
    놓친다. 실측 사례: 전체구조 목표 150 이 잔차 7e-4 로 1000 반복을 마쳤지만
    Cd 4.09, Cl 2.56(대칭이라 0 이어야 함)으로 미수렴이었다.

    판정: 마지막 window(기본 20%) 구간에서
      drift  = (후반 절반 평균 − 전반 절반 평균) / |전체 평균|   ← 표류
      osc    = 표준편차 / |평균|                                  ← 진동
    반환: verdict(converged/drifting/oscillating/insufficient), drift, osc,
          hit_iteration_cap(반복 상한에서 끝났는지), Cd/Cl 평균
    """
    out: Dict[str, Any] = {"verdict": "insufficient", "drift": None, "osc": None,
                           "n": 0, "hit_iteration_cap": None,
                           "mean_Cd": None, "mean_Cl": None}
    hist = read_force_history(case_dir)
    t, cd = hist.get("time") or [], hist.get("Cd") or []
    cl = hist.get("Cl") or []
    n = min(len(t), len(cd))
    if n < 20:
        return out
    k = max(10, int(n * window))
    seg_cd = cd[-k:]
    seg_cl = cl[-k:] if len(cl) >= k else []
    half = len(seg_cd) // 2
    m = sum(seg_cd) / len(seg_cd)
    out["n"] = len(seg_cd)
    out["mean_Cd"] = m
    if seg_cl:
        out["mean_Cl"] = sum(seg_cl) / len(seg_cl)
    if abs(m) < 1e-12:
        return out
    m1 = sum(seg_cd[:half]) / max(1, half)
    m2 = sum(seg_cd[half:]) / max(1, len(seg_cd) - half)
    out["drift"] = (m2 - m1) / abs(m)
    var = sum((v - m) ** 2 for v in seg_cd) / len(seg_cd)
    out["osc"] = math.sqrt(var) / abs(m)

    # 반복 상한에서 끝났는지 — controlDict endTime 과 마지막 시간 비교
    try:
        ctrl = (case_dir / "system" / "controlDict").read_text()
        mm = re.search(r"^\s*endTime\s+\$?([0-9.eE+-]+)\s*;", ctrl, re.M)
        if not mm:
            mm = re.search(r"^endTimeValue\s+([0-9.eE+-]+)\s*;", ctrl, re.M)
        if mm:
            out["end_time"] = float(mm.group(1))
            out["hit_iteration_cap"] = t[-1] >= float(mm.group(1)) - 0.5
    except Exception:
        pass

    if abs(out["drift"]) > 0.02:
        out["verdict"] = "drifting"
    elif out["osc"] > 0.02:
        out["verdict"] = "oscillating"
    else:
        out["verdict"] = "converged"
    return out


# ── 결과 신뢰성 등급 (요구서 §18) ─────────────────────────────────────────
RELIABILITY_LABELS = {"green": "GREEN — 주요 검사 통과",
                      "yellow": "YELLOW — 주의가 필요한 항목 존재",
                      "red": "RED — 결과 신뢰성에 중대한 문제 가능성"}


def result_reliability(case_dir: Path,
                       adequacy: Optional[Dict[str, Any]] = None,
                       mesh_independence: Optional[bool] = None,
                       reference_checked: Optional[bool] = None) -> Dict[str, Any]:
    """해석 결과를 Green/Yellow/Red 로 등급화하고 사유를 함께 낸다.

    'Success' 한 마디로 끝내지 않고, 무엇이 확인됐고 무엇이 미확인인지 남긴다.
    판정 재료는 이미 프로그램이 갖고 있는 것들이다 — 힘 계수 수렴, 격자 적정성,
    비정상성, 기준면적 기록, 격자 독립성·문헌 비교 수행 여부.
    """
    checks: List[Dict[str, str]] = []
    red: List[str] = []
    yellow: List[str] = []

    ref = read_case_reference(case_dir)
    if not ref.get("Aref_m2"):
        red.append("기준면적(Aref)을 확인할 수 없습니다 — CD·CL 값의 의미가 불명확합니다.")
        checks.append({"항목": "기준면적", "결과": "확인 불가"})
    else:
        checks.append({"항목": "기준면적", "결과": f"{ref['Aref_m2']:.6e} m²"})

    is_tr = (case_dir / "transient_meta.json").exists()
    if is_tr:
        st = compute_transient_stats(case_dir)
        if st.get("is_unsteady"):
            checks.append({"항목": "비정상성", "결과":
                           f"포착 ({st.get('unsteady_by') or 'Cd 변동'})"})
        else:
            yellow.append("비정상 해석인데 진동이 잡히지 않았습니다 — URANS 가 와류를 "
                          "감쇠시켰거나 후류 격자가 부족할 수 있습니다.")
            checks.append({"항목": "비정상성", "결과": "미포착"})
        if st.get("n_samples", 0) < 200:
            yellow.append("시간평균 표본이 적습니다 — 적분 시간을 늘리십시오.")
    else:
        conv = steady_force_convergence(case_dir)
        v = conv["verdict"]
        if v == "drifting":
            red.append(f"힘 계수가 아직 표류 중입니다(후반 구간 변화 "
                       f"{conv['drift']*100:+.1f}%). 반복을 더 돌려야 합니다.")
        elif v == "oscillating":
            yellow.append(f"힘 계수가 진동합니다(변동 {conv['osc']*100:.1f}%). 정상상태 "
                          "결과로 단정하지 말고 비정상 해석 또는 시간평균 검토가 "
                          "필요할 수 있습니다.")
        elif v == "insufficient":
            yellow.append("힘 계수 이력이 짧아 수렴 여부를 판정하지 못했습니다.")
        if conv.get("hit_iteration_cap"):
            yellow.append("수렴 기준이 아니라 반복 상한에서 종료됐습니다.")
        checks.append({"항목": "힘 계수 수렴", "결과":
                       {"converged": "수렴", "drifting": "표류", "oscillating": "진동",
                        "insufficient": "판정 불가"}[v]})

    if adequacy:
        if not adequacy.get("surface_ok", True):
            red.append(f"표면 격자가 부족합니다(임계 치수당 "
                       f"{adequacy.get('surface_cells', 0):.1f}셀, 목표 "
                       f"{SURF_CELLS_TARGET:.0f}셀). 정밀화 레벨을 올리십시오.")
        if adequacy.get("needs_wake") and not adequacy.get("wake_ok", True):
            yellow.append(f"후류 격자가 부족합니다({adequacy.get('wake_cells', 0):.1f}셀, "
                          f"목표 {WAKE_CELLS_TARGET:.0f}셀) — DES/LES 가 사실상 RANS 로 "
                          "동작할 수 있습니다.")
        checks.append({"항목": "격자 적정성",
                       "결과": "충분" if adequacy.get("ok") else "부족"})

    if mesh_independence is False or mesh_independence is None:
        yellow.append("격자 독립성 검증이 확인되지 않았습니다.")
    checks.append({"항목": "격자 독립성",
                   "결과": "확인됨" if mesh_independence else "미실시/미확인"})
    if not reference_checked:
        yellow.append("문헌·실험값과의 비교가 확인되지 않았습니다.")
    checks.append({"항목": "문헌 비교",
                   "결과": "수행" if reference_checked else "미실시/미확인"})

    grade = "red" if red else ("yellow" if yellow else "green")
    return {"grade": grade, "label": RELIABILITY_LABELS[grade],
            "red": red, "yellow": yellow, "checks": checks,
            "is_transient": is_tr}


# ── 실행 전 사전 점검 (요구서 §24·§25) ────────────────────────────────────
def preflight_checks(mode: str, stl_path: Optional[Path], speed: float,
                     rho: float, nu: float, aref: Optional[float] = None,
                     est_cells: Optional[float] = None,
                     est_mem_gb: Optional[float] = None,
                     mem_limit_gb: Optional[float] = None,
                     turbulence: Optional[str] = None,
                     solver: str = "Steady") -> List[Dict[str, str]]:
    """해석 시작 전에 잘못된 설정을 잡아낸다.

    반환 항목마다 무엇이/왜/어떻게 세 가지를 담는다(요구서 §25).
    level='critical' 이면 실행을 막아야 하고, 'warning' 은 진행 가능하다.
    """
    out: List[Dict[str, str]] = []

    def add(level, what, why_, how):
        out.append({"level": level, "what": what, "why": why_, "how": how})

    if speed is None or speed <= 0:
        add("critical", "유속이 0 이하입니다.",
            "유속이 0 이면 유동이 없어 항력·양력이 정의되지 않습니다.",
            "유속을 0 보다 큰 값으로 입력하십시오.")
    if rho is None or rho <= 0:
        add("critical", "밀도가 0 이하입니다.",
            "밀도는 힘을 무차원화하는 기준이라 0 이하일 수 없습니다.",
            "물리 조건에서 밀도를 양수로 입력하십시오.")
    if nu is None or nu <= 0:
        add("critical", "동점성계수가 0 이하입니다.",
            "점성이 0 이하이면 Reynolds 수와 경계층이 정의되지 않습니다.",
            "물리 조건에서 동점성계수를 양수로 입력하십시오.")
    if aref is not None and aref <= 0:
        add("critical", "기준면적이 0 이하입니다.",
            "CD·CL 을 계산하려면 기준면적이 필요합니다.",
            "자동 계산을 사용하거나 기준면적을 직접 입력하십시오.")

    if stl_path is not None:
        if not Path(stl_path).exists():
            add("critical", "STL 파일을 찾을 수 없습니다.",
                "형상이 없으면 격자를 만들 수 없습니다.",
                "STL 을 다시 업로드하십시오.")
        else:
            f = stl_geometry_features(Path(stl_path))
            if not f.get("ok"):
                add("critical", "STL 을 읽지 못했습니다.",
                    f.get("error", ""), "다른 STL 로 다시 시도하십시오.")
            else:
                spans = f["spans"]
                if max(spans) <= 0:
                    add("critical", "형상 크기가 0 입니다.",
                        "바운딩박스가 비어 있어 도메인을 만들 수 없습니다.",
                        "STL 내보내기 설정을 확인하십시오.")
                else:
                    if max(spans) > 1e5 or max(spans) < 0.1:
                        add("warning", f"형상 크기가 비정상적으로 보입니다"
                            f"(최대변 {max(spans):.3g} mm).",
                            "STL 단위가 mm 가 아닐 수 있습니다. 프로그램은 mm 로 "
                            "가정합니다.",
                            "STL 단위를 확인하고 필요하면 mm 로 다시 내보내십시오.")
                    if f["n_open_edges"] > 0:
                        add("warning", f"STL 표면에 열린 경계가 "
                            f"{f['n_open_edges']}개 있습니다.",
                            "닫히지 않은 표면은 내부·외부 구분이 모호해 격자 생성이 "
                            "실패하거나 기준면적 자동 계산이 절반이 될 수 있습니다.",
                            "닫힌 형상이어야 한다면 STL 을 수정하고, 얇은 판재라면 "
                            "기준면적을 직접 입력하십시오.")
                    if f["n_nonmanifold_edges"] > 0:
                        add("warning", f"non-manifold 에지가 "
                            f"{f['n_nonmanifold_edges']}개 있습니다.",
                            "한 에지를 3개 이상의 면이 공유하면 snappyHexMesh 가 "
                            "표면을 잘못 인식할 수 있습니다.",
                            "CAD 에서 형상을 정리한 뒤 다시 내보내십시오.")
                    if f["n_degenerate"] > 0:
                        add("warning", f"면적이 0 인 삼각형이 "
                            f"{f['n_degenerate']}개 있습니다.",
                            "퇴화 삼각형은 표면 인식과 격자 품질을 해칩니다.",
                            "STL 을 정리(cleanup)한 뒤 다시 내보내십시오.")

    if est_cells is not None:
        if est_cells > 3.0e7:
            add("critical", f"예상 셀 수가 너무 많습니다(약 {est_cells/1e6:.0f}백만).",
                "이 규모는 현재 장비에서 격자 생성 단계부터 실패하거나 며칠이 "
                "걸립니다.",
                "정밀화 레벨을 낮추거나 배경격자 목표 셀 수를 줄이십시오.")
        elif est_cells > 1.0e7:
            add("warning", f"예상 셀 수가 많습니다(약 {est_cells/1e6:.0f}백만).",
                "격자 생성과 해석에 수 시간 이상이 걸릴 수 있습니다.",
                "먼저 낮은 레벨로 경향을 확인한 뒤 올리는 편이 안전합니다.")
        elif est_cells < 5000:
            add("warning", f"예상 셀 수가 매우 적습니다(약 {est_cells:,.0f}개).",
                "형상을 해상하지 못해 힘 계수가 크게 어긋날 수 있습니다.",
                "정밀화 레벨을 올리거나 배경격자를 촘촘히 하십시오.")
    if est_mem_gb and mem_limit_gb and est_mem_gb > mem_limit_gb * 0.9:
        add("critical", f"예상 메모리({est_mem_gb:.1f} GB)가 가용 메모리"
            f"({mem_limit_gb:.1f} GB)에 근접합니다.",
            "격자 생성 중 메모리가 부족하면 프로세스가 강제 종료됩니다.",
            "정밀화 레벨을 낮추거나 목표 셀 수를 줄이십시오.")

    if solver == "Steady" and turbulence and turbulence != "kOmegaSST":
        add("critical", f"정상 해석에는 {turbulence} 를 쓸 수 없습니다.",
            "DES/LES 계열은 시간 전진이 전제입니다.",
            "Solver 를 Transient 로 바꾸거나 난류모델을 kOmegaSST 로 되돌리십시오.")
    return out


def patch_fluid_properties(case_dir: Path, rho: float, nu: float) -> Dict[str, Any]:
    """사용자가 지정한 유체 물성을 케이스에 반영한다.

    종전에는 UI 가 밀도 ρ 와 동점성계수 ν 를 입력받고도 케이스에 전달하지 않아,
    템플릿 값(nu 1.19e-6, rhoInf 1025)이 항상 쓰였다. ν 를 바꿔도 Reynolds 수와
    Cd 가 달라지지 않는 상태였다.

    - constant/transportProperties : nu      (동점성계수 [m²/s])
    - system/controlDict           : rhoInf  (forceCoeffs·forces 의 기준 밀도)

    값이 파일의 기존 값과 같으면 파일을 건드리지 않는다(기본값 경로의 산출물이
    종전과 바이트 단위로 같아야 하기 때문이다).
    반환: 실제로 바뀐 항목과 값.
    """
    out: Dict[str, Any] = {"nu": None, "rhoInf": None, "changed": []}

    def _same(a: float, b: float) -> bool:
        return abs(a - b) <= max(1e-12, abs(b) * 1e-9)

    tp = case_dir / "constant" / "transportProperties"
    if tp.exists() and nu and nu > 0:
        t = tp.read_text(encoding="utf-8")
        m = re.search(r"^(\s*nu\s+)([0-9.eE+-]+)(\s*;)", t, re.M)
        if m:
            out["nu"] = float(m.group(2))
            if not _same(float(nu), float(m.group(2))):
                t = t[:m.start()] + f"{m.group(1)}{nu:.6g}{m.group(3)}" + t[m.end():]
                tp.write_text(t, encoding="utf-8")
                out["nu"] = float(nu)
                out["changed"].append("nu")

    ctrl = case_dir / "system" / "controlDict"
    if ctrl.exists() and rho and rho > 0:
        t = ctrl.read_text(encoding="utf-8")
        vals = [float(x) for x in re.findall(r"^\s*rhoInf\s+([0-9.eE+-]+)\s*;", t, re.M)]
        if vals:
            out["rhoInf"] = vals[0]
            if not all(_same(float(rho), v) for v in vals):
                t = re.sub(r"^(\s*rhoInf\s+)[0-9.eE+-]+(\s*;)",
                           lambda mm: f"{mm.group(1)}{rho:g}{mm.group(2)}", t, flags=re.M)
                ctrl.write_text(t, encoding="utf-8")
                out["rhoInf"] = float(rho)
                out["changed"].append("rhoInf")
    return out


def read_case_reference(case_dir: Path) -> Dict[str, Optional[float]]:
    """케이스가 실제로 사용한 기준값을 controlDict 에서 읽는다.

    결과 파일에는 '계산에 실제로 쓰인' Aref·lRef·rhoInf·magUInf 가 남아야 한다.
    UI 설정이 아니라 케이스 파일을 읽는 이유는, 자동 계산·사용자 입력·재실행이
    섞여도 산출물과 항상 일치시키기 위해서다.
    """
    out: Dict[str, Optional[float]] = {"Aref_m2": None, "lRef_m": None,
                                       "rhoInf": None, "magUInf": None}
    ctrl = case_dir / "system" / "controlDict"
    if not ctrl.exists():
        return out
    try:
        t = ctrl.read_text(encoding="utf-8")
    except Exception:
        return out
    for key, pat in (("Aref_m2", r"^\s*Aref\s+([0-9.eE+-]+)\s*;"),
                     ("lRef_m", r"^\s*lRef\s+([0-9.eE+-]+)\s*;"),
                     ("rhoInf", r"^\s*rhoInf\s+([0-9.eE+-]+)\s*;"),
                     ("magUInf", r"^\s*magUInf\s+([0-9.eE+-]+)\s*;")):
        m = re.search(pat, t, re.M)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def read_stl_triangles(stl_path: Path) -> List[Tuple[Tuple[float, float, float], ...]]:
    """STL(ASCII/binary)을 읽어 삼각형 꼭짓점 목록을 반환.

    반환: [((x0,y0,z0),(x1,y1,z1),(x2,y2,z2)), ...]  (좌표 단위는 STL 그대로 = mm 가정)
    """
    import struct as _struct

    tris: List[Tuple[Tuple[float, float, float], ...]] = []

    with open(stl_path, "rb") as f:
        header = f.read(80)
    is_ascii = header.lstrip()[:5].lower().startswith(b"solid")

    if is_ascii:
        text = stl_path.read_text(errors="ignore")
        verts = [
            (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            for m in re.finditer(
                r"vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)", text)
        ]
        # ASCII는 vertex가 3개씩 한 삼각형
        for i in range(0, len(verts) - 2, 3):
            tris.append((verts[i], verts[i + 1], verts[i + 2]))
        # 일부 binary STL은 헤더가 'solid'로 시작 → ASCII 파싱 실패 시 binary 재시도
        if tris:
            return tris

    with open(stl_path, "rb") as f:
        f.read(80)
        raw = f.read(4)
        if len(raw) < 4:
            return tris
        n_tri = _struct.unpack("<I", raw)[0]
        for _ in range(n_tri):
            chunk = f.read(50)  # 12(normal)+36(3 verts)+2(attr)
            if len(chunk) < 50:
                break
            p0 = _struct.unpack("<fff", chunk[12:24])
            p1 = _struct.unpack("<fff", chunk[24:36])
            p2 = _struct.unpack("<fff", chunk[36:48])
            tris.append((p0, p1, p2))
    return tris


def compute_projected_area(stl_path: Path, flow_dir: Tuple[float, float, float]) -> float:
    """STL 형상의 실제 정면 투영 면적[m²]을 유동 방향 기준으로 계산.

    각 삼각형의 면적벡터 A_i·n_i 를 유동 단위벡터 d 에 투영한 절댓값의 합을
    2로 나눈다(닫힌 표면은 앞면+뒷면이 같은 실루엣을 이루므로 ÷2).
    → 그물실 트와인의 정면 투영(실루엣) 면적. 영각이 바뀌면 d 가 바뀌어 자동 변화.

    STL 좌표는 mm 단위로 가정하고 결과는 m² 로 환산해 반환한다.
    """
    import math as _math

    dn = _math.sqrt(sum(c * c for c in flow_dir))
    if dn == 0:
        d = (1.0, 0.0, 0.0)
    else:
        d = (flow_dir[0] / dn, flow_dir[1] / dn, flow_dir[2] / dn)

    tris = read_stl_triangles(stl_path)
    if not tris:
        return 0.0

    total = 0.0  # mm²
    for p0, p1, p2 in tris:
        e1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        e2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        # 면적벡터 = 0.5 × (e1 × e2)  (크기 = 삼각형 면적, 방향 = 법선)
        cx = 0.5 * (e1[1] * e2[2] - e1[2] * e2[1])
        cy = 0.5 * (e1[2] * e2[0] - e1[0] * e2[2])
        cz = 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        total += abs(cx * d[0] + cy * d[1] + cz * d[2])

    # 닫힌 표면: 앞/뒤면이 합쳐져 2배 → ÷2, mm²→m²: ÷1e6
    return (total * 0.5) / 1.0e6


# ── 그물실 격자 해상도 ────────────────────────────────────────────────────
# 배경격자는 '셀 개수'가 아니라 형상 크기 기준으로 정해지므로, 망목(a)이 커지면
# 그물실(지름 d) 대비 격자가 조용히 거칠어진다. 예: 단위셀 base=a/16 이면
#   a=20mm  → 최소셀 0.25mm → 실 지름당 12셀 (양호)
#   a=124mm → 최소셀 0.97mm → 실 지름당  3셀 (부족)
# 원통 표면의 경계층·박리를 풀려면 통상 지름당 10셀 이상이 필요하다.
TWINE_CELLS_TARGET = 10.0


def net_grid_base_cell(crit_mm: float, level: int, target: float,
                       domain_m: Optional[Tuple[float, float, float]] = None,
                       max_base_cells: int = 3_000_000) -> float:
    """배경격자 재설계의 배경 셀 크기[m].

    빌더와 UI 판정이 같은 식을 쓰도록 분리한 것이며, 계산 내용은 종전 빌더
    코드와 동일하다. domain_m 을 주면 배경 셀 총수 상한까지 반영한다.
    """
    cell = (float(crit_mm) / 1000.0) * (2 ** int(level)) / float(target)
    cap_cell = float(crit_mm) / 1000.0
    capped = cell > cap_cell
    if capped:
        cell = cap_cell
    if domain_m:
        for _ in range(12):
            n = (domain_m[0] / cell) * (domain_m[1] / cell) * (domain_m[2] / cell)
            if n <= max_base_cells:
                break
            cell *= 1.26
    return cell


def twine_resolution(base_mm: float, wire_d_mm: float,
                     refine_level: int,
                     target: float = TWINE_CELLS_TARGET) -> Dict[str, Any]:
    """배경격자 크기와 그물실 지름으로 표면 격자 해상도를 평가한다.

    반환: finest_mm(최소 셀), cells_per_d(실 지름당 셀 수),
          required_level(target 을 만족하는 최소 정밀화 레벨), ok(bool)
    """
    base_mm = max(float(base_mm), 1e-9)
    lvl = max(0, int(refine_level))
    finest = base_mm / (2 ** lvl)
    cpd = (float(wire_d_mm) / finest) if finest > 0 else 0.0
    req = lvl
    if wire_d_mm > 0:
        # base/2^L <= d/target  →  2^L >= base*target/d
        need = base_mm * float(target) / float(wire_d_mm)
        req = max(lvl, int(math.ceil(math.log2(need))) if need > 1 else 0)
    return {"base_mm": base_mm, "finest_mm": finest, "cells_per_d": cpd,
            "required_level": req, "ok": cpd >= target, "target": target}


# ── 임계 최소 치수와 격자 적정성 ──────────────────────────────────────────
# 격자가 반드시 해상해야 하는 치수는 '형상 전체 크기'가 아니라 '가장 가는 부분'
# 이다(그물실 지름, 판재 두께). 이 값을 기준으로 두 가지를 따로 판정해야 한다.
#   ① 표면 정밀화 : 경계층·박리를 풀려면 임계치수당 10셀 이상
#   ② 후류 박스   : DES/LES 가 LES 모드로 전환하려면 후류 셀이 임계치수보다
#                   충분히 작아야 한다(임계치수당 5셀 이상 권장)
# 실측 사례: 3by3 그물(실 3mm)에서 표면은 12.4셀로 충분했지만 후류 박스가
# 레벨 2(셀 3.88mm > 실 3mm)라 DDES 가 RANS 로 동작해 비정상성을 못 잡았다.
SURF_CELLS_TARGET = 10.0     # 표면(경계층) 목표
WAKE_CELLS_TARGET = 5.0      # 후류(DES/LES) 목표


def critical_dimension(stl_path: Path) -> Dict[str, Any]:
    """STL 에서 '격자가 반드시 해상해야 할 임계 최소 치수'[mm] 를 자동 산정한다.

    반환: value(권장값), bbox_min(바운딩박스 최소변), thickness(닫힌 형상의
          체적/표면적 기반 두께 추정), closed, spans
    """
    out: Dict[str, Any] = {"value": 0.0, "bbox_min": 0.0, "thickness": None,
                           "closed": None, "spans": None, "basis": ""}
    tris = read_stl_triangles(stl_path)
    if not tris:
        out["basis"] = "STL 을 읽지 못했습니다"
        return out
    pts = [v for t in tris for v in t]
    spans = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)]
    out["spans"] = spans
    pos = [s for s in spans if s > 0]
    out["bbox_min"] = min(pos) if pos else 0.0

    # 닫힌 형상이면 체적/표면적으로 대표 두께를 함께 추정한다(참고용).
    try:
        closed = is_closed_surface(stl_path)
        out["closed"] = closed
        if closed:
            vol = 0.0
            for p0, p1, p2 in tris:
                vol += (p0[0]*(p1[1]*p2[2]-p1[2]*p2[1])
                        - p0[1]*(p1[0]*p2[2]-p1[2]*p2[0])
                        + p0[2]*(p1[0]*p2[1]-p1[1]*p2[0])) / 6.0
            area_mm2 = compute_surface_area(stl_path) * 1.0e6
            if area_mm2 > 0:
                out["thickness"] = abs(vol) * 6.0 / area_mm2   # 6V/A ≈ 원통 지름
    except Exception:
        pass

    out["value"] = out["bbox_min"]
    out["basis"] = "바운딩박스 최소변"
    return out


# ── STL 유형 자동 분류 (요구서 §3) ────────────────────────────────────────
# 형상마다 적절한 해석 설정이 다르므로, 먼저 무엇인지 알아야 권장값을 낼 수 있다.
# 판별에 쓰는 지표는 모두 STL 삼각형만으로 계산되며, 실측 4종(구·원기둥·카이트·
# 그물)에서 아래와 같이 뚜렷이 갈린다.
#
#            중심반경 CV   축반경 CV(최소)   두께비   채움률   표면적/투영
#   구           0.000        0.118         1.000    0.785      4.00
#   원기둥       —            0.000         0.250    1.000      3.53
#   카이트       0.361        0.387         0.196    0.402      2.35
#   그물         0.140        0.146         0.075    0.201      3.21
#
# 채움률 = (얇은 축 방향 투영면적) / (그 축에 수직인 바운딩박스 단면적)
#   → 그물은 구멍이 대부분이라 0.2, 판재는 0.4 이상.
# 표면적/투영 → 판재(양면) ≈ 2, 원통 다발 ≈ π, 구 = 4.
STL_TYPE_LABELS: Dict[str, str] = {
    "sphere":    "구 (Sphere)",
    "cylinder":  "원기둥 (Cylinder)",
    "kite":      "카이트·판재 (얇은 곡면)",
    "net_panel": "그물패널 (Net panel)",
    "complex":   "복잡한 입체물 (Complex 3D solid)",
    "unknown":   "자동 분류 불가",
}


def stl_geometry_features(stl_path: Path) -> Dict[str, Any]:
    """STL 한 번만 읽어 분류·검사에 쓰는 기하 지표를 모두 계산한다."""
    out: Dict[str, Any] = {"ok": False}
    tris = read_stl_triangles(stl_path)
    if not tris:
        out["error"] = "STL 을 읽지 못했습니다"
        return out

    pts = [v for t in tris for v in t]
    mn = [min(p[i] for p in pts) for i in range(3)]
    mx = [max(p[i] for p in pts) for i in range(3)]
    spans = [mx[i] - mn[i] for i in range(3)]

    area = 0.0
    vol = 0.0
    proj = [0.0, 0.0, 0.0]
    cents: List[Tuple[float, float, float]] = []
    edges: Dict[Tuple, int] = {}
    degenerate = 0
    for p0, p1, p2 in tris:
        e1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        e2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
        cx = 0.5*(e1[1]*e2[2] - e1[2]*e2[1])
        cy = 0.5*(e1[2]*e2[0] - e1[0]*e2[2])
        cz = 0.5*(e1[0]*e2[1] - e1[1]*e2[0])
        a = math.sqrt(cx*cx + cy*cy + cz*cz)
        area += a
        if a <= 1e-12:
            degenerate += 1
        for i, c in enumerate((cx, cy, cz)):
            proj[i] += abs(c)
        vol += (p0[0]*(p1[1]*p2[2]-p1[2]*p2[1])
                - p0[1]*(p1[0]*p2[2]-p1[2]*p2[0])
                + p0[2]*(p1[0]*p2[1]-p1[1]*p2[0])) / 6.0
        cents.append(((p0[0]+p1[0]+p2[0])/3.0,
                      (p0[1]+p1[1]+p2[1])/3.0,
                      (p0[2]+p1[2]+p2[2])/3.0))
        vs = [tuple(round(c, 4) for c in v) for v in (p0, p1, p2)]
        for i in range(3):
            key = tuple(sorted((vs[i], vs[(i+1) % 3])))
            edges[key] = edges.get(key, 0) + 1

    proj = [p * 0.5 for p in proj]          # 닫힌 표면 앞/뒷면 중복 제거
    vol = abs(vol)
    n_open = sum(1 for n in edges.values() if n == 1)
    n_nonmanifold = sum(1 for n in edges.values() if n > 2)

    cx0 = sum(c[0] for c in cents) / len(cents)
    cy0 = sum(c[1] for c in cents) / len(cents)
    cz0 = sum(c[2] for c in cents) / len(cents)

    def _cv(vals: List[float]) -> float:
        if not vals:
            return 9.0
        m = sum(vals) / len(vals)
        if m <= 0:
            return 9.0
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        return math.sqrt(v) / m

    r_cv = _cv([math.dist(c, (cx0, cy0, cz0)) for c in cents])

    axis_cv = []
    ctr = (cx0, cy0, cz0)
    for ax in range(3):
        i, j = [k for k in range(3) if k != ax]
        lo = mn[ax] + 0.1 * spans[ax]
        hi = mx[ax] - 0.1 * spans[ax]
        # 원기둥 끝면(캡)은 반경이 작아 분산을 키우므로 중앙 80% 구간만 본다
        sel = [c for c in cents if lo <= c[ax] <= hi]
        if len(sel) < 10:
            sel = cents
        axis_cv.append(_cv([math.hypot(c[i]-ctr[i], c[j]-ctr[j]) for c in sel]))

    thin = min(range(3), key=lambda i: spans[i])
    face = [spans[i] for i in range(3) if i != thin]
    face_area = face[0] * face[1]
    fill = (proj[thin] / face_area) if face_area > 0 else 0.0
    max_span = max(spans) or 1.0

    out.update({
        "ok": True, "n_tri": len(tris), "spans": spans, "bbox_min": mn, "bbox_max": mx,
        "area_mm2": area, "volume_mm3": vol, "closed": (n_open == 0 and n_nonmanifold == 0),
        "n_open_edges": n_open, "n_nonmanifold_edges": n_nonmanifold,
        "n_degenerate": degenerate,
        "sphericity": ((36*math.pi*vol**2) / area**3) ** (1/3) if area > 0 and vol > 0 else 0.0,
        "r_cv": r_cv, "axis_cv": axis_cv, "thin_axis": thin,
        "thin_ratio": spans[thin] / max_span, "fill_ratio": fill,
        "proj_mm2": proj, "area_per_proj": (area / proj[thin]) if proj[thin] > 0 else 0.0,
    })
    return out


def classify_stl(stl_path: Path) -> Dict[str, Any]:
    """STL 유형을 자동 분류한다.

    반환: type, label, confidence(0~1), basis(근거 문장), needs_confirm, features
    확신이 낮으면(<0.7) type='unknown' 으로 두고 사용자 확인을 요구한다(요구서 §3).
    임의로 하나를 골라 진행하지 않는다.
    """
    f = stl_geometry_features(stl_path)
    out: Dict[str, Any] = {"type": "unknown", "confidence": 0.0, "basis": "",
                           "features": f, "needs_confirm": True}
    if not f.get("ok"):
        out["basis"] = f.get("error", "STL 을 읽지 못했습니다")
        out["label"] = STL_TYPE_LABELS["unknown"]
        return out

    r_cv = f["r_cv"]
    ax_cv = min(f["axis_cv"])
    thin_r = f["thin_ratio"]
    fill = f["fill_ratio"]
    app = f["area_per_proj"]

    if r_cv < 0.05 and f["sphericity"] > 0.85:
        out["type"] = "sphere"
        out["confidence"] = 0.98 if r_cv < 0.02 else 0.85
        out["basis"] = (f"중심에서 표면까지의 거리가 거의 일정합니다"
                        f"(변동계수 {r_cv:.3f}, 구형도 {f['sphericity']:.3f}).")
    elif ax_cv < 0.06:
        out["type"] = "cylinder"
        out["confidence"] = 0.95 if ax_cv < 0.02 else 0.80
        out["basis"] = (f"한 축(#{f['axis_cv'].index(ax_cv)+1}) 주위의 반지름이 거의 "
                        f"일정합니다(변동계수 {ax_cv:.3f}).")
    elif thin_r < 0.20 and fill < 0.32:
        out["type"] = "net_panel"
        out["confidence"] = 0.90 if fill < 0.25 else 0.70
        out["basis"] = (f"두께가 매우 얇고(최대변의 {thin_r*100:.1f}%) 투영면의 "
                        f"{fill*100:.0f}% 만 채워져 있어 구멍이 대부분입니다. "
                        f"표면적/투영면적 = {app:.2f} 로 원통(π≈3.14) 다발에 가깝습니다.")
    elif thin_r < 0.30 and app < 2.8:
        out["type"] = "kite"
        out["confidence"] = 0.85 if fill > 0.35 else 0.65
        out["basis"] = (f"두께가 얇고(최대변의 {thin_r*100:.1f}%) 표면적/투영면적 = "
                        f"{app:.2f} 로 양면 판재(≈2)에 가깝습니다.")
    else:
        out["type"] = "complex"
        out["confidence"] = 0.55
        out["basis"] = ("구·원기둥·판재·그물 중 어느 특징도 뚜렷하지 않습니다"
                        f"(중심반경 CV {r_cv:.3f}, 축반경 CV {ax_cv:.3f}, "
                        f"두께비 {thin_r:.3f}, 채움률 {fill:.3f}).")

    out["needs_confirm"] = out["confidence"] < 0.70
    if out["needs_confirm"]:
        out["type_guess"] = out["type"]
        out["type"] = "unknown"
    out["label"] = STL_TYPE_LABELS[out["type"]]
    return out


# ── STL 유형별 권장 설정 (요구서 §4·§13) ──────────────────────────────────
# 값은 '권장'이며 강제하지 않는다(요구서 §27). 근거를 함께 담아 UI 가 '왜?' 로
# 펼쳐 보일 수 있게 한다.
STL_TYPE_PRESETS: Dict[str, Dict[str, Any]] = {
    "sphere": {
        "analysis_mode": "full_structure", "solver": "Steady",
        "refine_level": 3, "auto_refine": True, "aref_mode": "자동",
        "solver_reason":
            "박리점이 표면 위를 움직이는 형상이라 정상 RANS 가 후류를 과소평가합니다. "
            "빠른 확인은 Steady 로 하되, 항력을 정량적으로 쓰려면 Transient(DDES)로 "
            "재검증하십시오.",
        "warnings": [
            "본 프로그램 실측: 구(Re=2.5e5)에서 simpleFoam Cd=0.228, DDES Cd=0.218 로 "
            "실험값 0.5 의 절반 수준입니다. 구는 정상 RANS 검증에 불리한 형상입니다.",
        ],
    },
    "cylinder": {
        "analysis_mode": "full_structure", "solver": "Transient",
        "refine_level": 4, "auto_refine": True, "aref_mode": "자동",
        "solver_reason":
            "원기둥 후방에서 주기적인 와류 방출이 발생할 가능성이 있으므로 시간에 따른 "
            "유동 변화를 계산하는 편이 적합합니다. 다만 조건에 따라 다르므로 강제하지 "
            "않습니다.",
        "warnings": [
            "비정상 해석은 후류 격자가 충분해야 의미가 있습니다. 격자 적정성 판정에서 "
            "후류 셀 수를 확인하십시오.",
        ],
    },
    "kite": {
        "analysis_mode": "full_structure", "solver": "Steady",
        "refine_level": 3, "auto_refine": True, "aref_mode": "직접 입력",
        "solver_reason":
            "받음각이 작아 유동이 붙어 있으면 Steady 로 충분하고, 큰 받음각에서 대규모 "
            "박리가 생기면 Transient 가 필요합니다. 받음각 조건에 따라 선택하십시오.",
        "warnings": [
            "열린 곡면이면 기준면적 자동 계산(닫힌 표면 가정)이 실제의 절반이 됩니다. "
            "기준면적을 직접 입력하십시오.",
            "받음각 40° 이상 구간은 대규모 박리라 정상 해석 결과를 보수적으로 보십시오.",
        ],
    },
    "net_panel": {
        "analysis_mode": "unit_cell", "solver": "Steady",
        "refine_level": 6, "auto_refine": True, "aref_mode": "자동",
        "net_grid_redesign": True, "net_grid_target_cells": 75.0,
        "solver_reason":
            "그물은 다수의 가는 실에 힘이 분산돼 개별 후류의 위상이 상쇄되므로 정상 "
            "해석으로도 항력이 잘 잡힙니다. 실측에서 정상해와 DDES 시간평균의 차이는 "
            "6% 였습니다.",
        "warnings": [
            "그물실 직경이 2~3 mm 인 실제 형상을 직접 해석하면 실 직경을 충분히 "
            "해상하기 위한 매우 작은 격자가 필요하며, 계산시간과 메모리 사용량이 크게 "
            "증가할 수 있습니다.",
            "실측: 실 지름당 40셀 Cd=0.776 → 75셀 0.936 으로 해상도에 크게 좌우됩니다. "
            "배경격자 재설계를 켜고 목표 셀 수를 확인하십시오.",
        ],
    },
    "complex": {
        "analysis_mode": "full_structure", "solver": "Steady",
        "refine_level": 4, "auto_refine": True, "aref_mode": "자동",
        "solver_reason":
            "복잡한 후류가 예상되므로 결과의 시간 변화를 확인하십시오. 정상 해석으로 "
            "먼저 경향을 보고, 힘 계수가 진동하면 Transient 로 재검증하는 순서를 "
            "권합니다.",
        "warnings": [
            "형상이 복잡하면 표면 정밀화 셀이 급증합니다. 예상 셀 수를 확인한 뒤 "
            "실행하십시오.",
        ],
    },
}


# ── 유체 프리셋 (요구서 §7) ───────────────────────────────────────────────
# 기본값은 종전 템플릿과 같은 해수 값이라, 프리셋을 쓰지 않으면 동작이 바뀌지 않는다.
FLUID_PRESETS: Dict[str, Dict[str, float]] = {
    "해수 (20℃)":  {"rho": 1025.0, "nu_e6": 1.19},
    "담수 (20℃)":  {"rho": 998.2,  "nu_e6": 1.004},
    "공기 (20℃)":  {"rho": 1.204,  "nu_e6": 15.11},
}


def estimate_mesh_size(background_cells: float, surface_area_m2: float,
                       finest_cell_m: float) -> Dict[str, Any]:
    """격자 생성 전에 최종 셀 수와 메모리를 개략 추정한다(요구서 §14).

    정밀화 셀은 표면 주위 껍질에 생기므로 (표면적 / 최소셀²) 에 비례한다.
    비례계수 C 는 본 프로그램 실측 2건으로 보정했다.
        목표 75  : 배경 418,996 → 최종 1,958,719 (C = 2.4)
        목표 150 : 배경 1,695,573 → 최종 6,465,852 (C = 3.0)
    메모리는 snappyHexMesh 최대 사용량 실측(1,189만 셀에서 19 GB)에서 환산했다.

    반환값은 범위다. 형상·정밀화 설정에 따라 달라지므로 단일값으로 쓰지 않는다.
    """
    out: Dict[str, Any] = {"ok": False}
    if finest_cell_m <= 0 or surface_area_m2 <= 0:
        return out
    shell = surface_area_m2 / (finest_cell_m ** 2)
    lo = background_cells + 2.4 * shell
    hi = background_cells + 3.0 * shell
    out.update({
        "ok": True,
        "background_cells": background_cells,
        "cells_min": lo, "cells_max": hi,
        # 실측 19 GB / 11.89e6 셀 = 1.6 GB/백만셀 (snappy 최대). 범위로 제시.
        "mem_min_gb": lo / 1e6 * 1.3,
        "mem_max_gb": hi / 1e6 * 1.9,
    })
    return out


def mesh_adequacy(critical_mm: float, base_mm: float,
                  surface_level: int, box_level: int,
                  model_family: str = "RAS") -> Dict[str, Any]:
    """임계 치수 대비 표면·후류 격자의 적정성을 판정하고 권고 레벨을 낸다."""
    base_mm = max(float(base_mm), 1e-9)
    crit = max(float(critical_mm), 1e-9)

    def _cells(level):
        return crit / (base_mm / (2 ** max(0, int(level))))

    def _need(target):
        # base/2^L <= crit/target  →  2^L >= base*target/crit
        r = base_mm * target / crit
        return max(0, int(math.ceil(math.log2(r)))) if r > 1 else 0

    surf_cells = _cells(surface_level)
    wake_cells = _cells(box_level)
    is_les = str(model_family).upper() in ("LES", "DES")
    out = {
        "critical_mm": crit, "base_mm": base_mm,
        "surface_level": int(surface_level), "box_level": int(box_level),
        "surface_cell_mm": base_mm / (2 ** max(0, int(surface_level))),
        "wake_cell_mm": base_mm / (2 ** max(0, int(box_level))),
        "surface_cells": surf_cells, "wake_cells": wake_cells,
        "surface_ok": surf_cells >= SURF_CELLS_TARGET,
        "wake_ok": (wake_cells >= WAKE_CELLS_TARGET) if is_les else True,
        "surface_required": _need(SURF_CELLS_TARGET),
        "wake_required": _need(WAKE_CELLS_TARGET) if is_les else int(box_level),
        "model_family": "LES/DES" if is_les else "RANS",
        "needs_wake": is_les,
    }
    out["ok"] = out["surface_ok"] and out["wake_ok"]
    return out


def mesh_adequacy_table(critical_mm: float, base_mm: float,
                        levels=(2, 3, 4, 5, 6, 7),
                        model_family: str = "RAS") -> List[Dict[str, Any]]:
    """레벨별 셀 크기와 판정을 표로 만들어 UI 에 그대로 보여주기 위한 헬퍼."""
    rows = []
    for lv in levels:
        cell = base_mm / (2 ** lv)
        rows.append({
            "level": lv,
            "cell_mm": cell,
            "cells": critical_mm / cell if cell > 0 else 0.0,
            "surface_ok": (critical_mm / cell) >= SURF_CELLS_TARGET if cell > 0 else False,
            "wake_ok": (critical_mm / cell) >= WAKE_CELLS_TARGET if cell > 0 else False,
        })
    return rows


def tighten_mesh_quality(snappy_path: Path) -> Dict[str, Any]:
    """snappyHexMeshDict 의 meshQualityControls 를 조여 슬리버 셀을 억제한다.

    체적은 정상인데 한 방향으로만 극단적으로 얇은 셀(슬리버)이 하나만 있어도
    비정상 해석의 시간간격이 붕괴한다(실측: Courant 297 → deltaT 2.7e-8 →
    743만 스텝, 실행 불가). 이런 셀은 체적이나 종횡비로는 잡히지 않으므로
    snappyHexMesh 가 애초에 만들지 않도록 품질 기준을 높인다.

    대가로 형상 스냅 품질(모서리 재현)이 다소 떨어질 수 있다.
    """
    if not snappy_path.exists():
        return {"ok": False, "changed": []}
    txt = snappy_path.read_text()
    # (키, 기존값 정규식, 새 값) — 첫 번째(주 meshQualityControls)만 바꾸고
    # addLayers 용 완화 블록(relaxed)은 건드리지 않는다.
    rules = [
        ("minTetQuality", r"minTetQuality\s+[0-9.eE+-]+;", "minTetQuality           1e-09;"),
        ("minDeterminant", r"minDeterminant\s+[0-9.eE+-]+;", "minDeterminant          0.01;"),
        ("minVolRatio",    r"minVolRatio\s+[0-9.eE+-]+;",    "minVolRatio             0.05;"),
        ("minTwist",       r"minTwist\s+[0-9.eE+-]+;",       "minTwist                0.05;"),
        ("minFaceWeight",  r"minFaceWeight\s+[0-9.eE+-]+;",  "minFaceWeight           0.10;"),
    ]
    changed = []
    for name, pat, rep in rules:
        txt, n = re.subn(pat, rep, txt, count=1)
        if n:
            changed.append(name)
    snappy_path.write_text(txt)
    logger.info(f"[Mesh] 격자 품질 기준 강화 — {', '.join(changed) or '변경 없음'}")
    return {"ok": bool(changed), "changed": changed}


def validate_unit_cell_stl(stl_path: Path) -> Dict[str, Any]:
    """단위셀 모드가 요구하는 STL 조건을 검사한다.

    단위셀 도메인은 '원점 중심, 한 변 a=max(x범위,y범위)인 정사각'으로 생성된다.
    STL 이 원점 중심이 아니거나 정사각이 아니면 형상 일부가 도메인 밖으로 나가
    snappyHexMesh 가 아무것도 잡지 못하고, 그 결과 Cd=0 이 조용히 기록된다.
    (실제로 net2_onemesh.stl 이 y=102mm 까지 뻗어 도메인 ±61mm 를 벗어났다.)

    반환: ok(bool), a_mm, issues(list[str]), bbox
    """
    out: Dict[str, Any] = {"ok": False, "issues": [], "a_mm": 0.0, "bbox": None}
    tris = read_stl_triangles(stl_path)
    if not tris:
        out["issues"].append("STL 을 읽지 못했습니다.")
        return out
    pts = [v for t in tris for v in t]
    bx = (min(p[0] for p in pts), max(p[0] for p in pts))
    by = (min(p[1] for p in pts), max(p[1] for p in pts))
    bz = (min(p[2] for p in pts), max(p[2] for p in pts))
    out["bbox"] = {"x": bx, "y": by, "z": bz}
    sx, sy = bx[1] - bx[0], by[1] - by[0]
    a = max(sx, sy)
    out["a_mm"] = a
    if a <= 0:
        out["issues"].append("XY 범위가 0 입니다.")
        return out
    tol = 0.05 * a
    if abs(bx[0] + bx[1]) > tol or abs(by[0] + by[1]) > tol:
        out["issues"].append(
            f"원점 중심이 아닙니다 (중심 x={(bx[0]+bx[1])/2:.1f}, "
            f"y={(by[0]+by[1])/2:.1f} mm). 도메인은 원점 중심 ±{a/2:.1f} mm 로 "
            f"만들어지므로 형상 일부가 도메인 밖으로 나갑니다.")
    if abs(sx - sy) > tol:
        out["issues"].append(
            f"정사각이 아닙니다 (x범위 {sx:.1f} mm, y범위 {sy:.1f} mm). "
            f"단위셀은 정사각 망목을 가정합니다.")
    out["ok"] = not out["issues"]
    return out


def unit_cell_base_mm(cell_size_m: float) -> float:
    """UnitCellCaseBuilder._patch_blockMesh 와 동일한 배경격자 산정식."""
    return max(2.0, float(cell_size_m) * 1000.0 / 16.0)


def is_closed_surface(stl_path: Path) -> bool:
    """STL 이 닫힌 매니폴드인지 판정한다(모든 에지가 정확히 2개 삼각형에 공유).

    그물실(트와인)은 닫힌 원통 → True. 카이트·돛·판재 같은 열린 단일 곡면 → False.
    compute_projected_area() 의 ÷2(앞/뒷면 중복 제거)는 닫힌 표면에서만 타당하므로,
    이 판정이 False 면 그 값은 실제 면적의 절반이 된다(UI 경고 근거).
    """
    tris = read_stl_triangles(stl_path)
    if not tris:
        return False
    edges: Dict[Tuple, int] = {}
    for tri in tris:
        vs = [tuple(round(c, 4) for c in v) for v in tri]
        for i in range(3):
            key = tuple(sorted((vs[i], vs[(i + 1) % 3])))
            edges[key] = edges.get(key, 0) + 1
    return all(n == 2 for n in edges.values())


def compute_surface_area(stl_path: Path) -> float:
    """STL 삼각형 면적의 단순 합[m²] — '면 자체의 면적'(투영 아님).

    카이트/돛처럼 열린 곡면의 기준면적으로 쓰인다. STL 좌표 mm 가정.
    """
    total = 0.0
    for p0, p1, p2 in read_stl_triangles(stl_path):
        e1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        e2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        cx = e1[1] * e2[2] - e1[2] * e2[1]
        cy = e1[2] * e2[0] - e1[0] * e2[2]
        cz = e1[0] * e2[1] - e1[1] * e2[0]
        total += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    return total / 1.0e6


def detect_stl_cell_size(stl_path: Path) -> Dict[str, float]:
    """STL 바운딩 박스에서 단위 셀 크기·와이어 직경·고형률 자동 감지.

    가정: STL 좌표 단위 = mm, XY 범위 = 단위 셀 크기, Z 범위 = 와이어 직경.
    반환값의 cell_size_mm 및 wire_diameter_mm 단위는 모두 mm.
    추가로 정면(X축, 영각 0°) 실제 투영면적 frontal_area_m2 를 함께 반환한다.
    """
    tris = read_stl_triangles(stl_path)
    if not tris:
        return {"cell_size_mm": 20.0, "wire_diameter_mm": 2.0,
                "solidity": 0.10, "frontal_area_m2": 0.0}

    xs = [v[0] for t in tris for v in t]
    ys = [v[1] for t in tris for v in t]
    zs = [v[2] for t in tris for v in t]

    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    z_span = max(zs) - min(zs)

    # XY 최대 범위 = 단위 셀 크기, Z 범위 = 와이어 직경
    cell_size_mm    = max(x_span, y_span)
    wire_diameter_mm = z_span

    # 고형률 추정: 사각망 근사 Sn = 2d/a (d=와이어직경, a=망목크기)
    solidity = min(0.95, 2.0 * wire_diameter_mm / cell_size_mm) if cell_size_mm > 0 else 0.10

    # 고정 기준면적용 투영면적: 그물면 법선(Z축, 정면) 방향 실측 투영면적
    # = 그물을 통해 들여다본 트와인 실루엣 면적 (영각 무관, Aref 기준)
    frontal_area_m2 = compute_projected_area(stl_path, (0.0, 0.0, 1.0))

    return {
        "cell_size_mm":     round(cell_size_mm, 3),
        "wire_diameter_mm": round(wire_diameter_mm, 3),
        "solidity":         round(solidity, 4),
        "frontal_area_m2":  frontal_area_m2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 비정상(transient) 해석 — pimpleFoam 지원
# ═══════════════════════════════════════════════════════════════════════════
#
# 설계 원칙(지시서 §2·§19): 기존 steady 케이스 생성 로직을 일절 건드리지 않는다.
# 빌더가 평소대로 케이스를 만든 '뒤에' apply_transient_settings() 로 사후 패치만
# 한다. 따라서 Solver=Steady 경로는 바이트 단위로 종전과 동일하다.
#
# OpenFOAM v2312 기준으로 확인한 필수 사항:
#  - fvSolution 의 solvers 에 p / "(U|k|omega)" 만 있어 PIMPLE 이 요구하는
#    pFinal·UFinal 등을 못 찾고 실행 실패한다 → 정규식을 Final 포함으로 넓힌다.
#  - fvSchemes 의 ddtSchemes 포맷이 모드마다 다르다(1줄 vs 여러 줄) → 정규식 처리.
#  - SIMPLE 블록을 PIMPLE 로 교체하고 완화계수를 transient 용으로 바꾼다.

TRANSIENT_DEFAULTS: Dict[str, Any] = {
    "end_time":        30.0,     # 물리시간 [s]
    "delta_t":         1.0e-3,   # 초기 시간간격 [s]
    "max_co":          0.8,      # 보완⑤: 원본 1.0 → 0.8 (nOuterCorrectors=1 은 PISO)
    "max_delta_t":     0.01,
    "write_interval":  0.5,      # adjustableRunTime [s]
    "n_outer":         1,
    "n_correctors":    2,
    "n_non_orth":      0,
    "turbulence":      "kOmegaSST",
    "ddt_scheme":      "backward",
}

# 지시서 보완②: URANS 가 비정상성을 억제할 때 넘어갈 대안. v2312 설치본에서
# 라이브러리에 등록되어 있음을 확인한 모델만 노출한다.
TRANSIENT_TURBULENCE_MODELS: Dict[str, str] = {
    "kOmegaSST":           "RAS",   # 기본값 — 기존 steady 와 동일
    "kOmegaSSTDDES":       "LES",
    "kOmegaSSTDES":        "LES",
    "SpalartAllmarasDDES": "LES",
}


def _sub_once(text: str, pattern: str, repl: str, flags=0) -> Tuple[str, bool]:
    """정규식 치환 + 실제로 바뀌었는지 여부를 함께 반환(조용한 실패 방지)."""
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    return new, bool(n)


def apply_transient_settings(case_dir: Path, **kw) -> Dict[str, Any]:
    """이미 생성된 steady 케이스를 pimpleFoam(비정상) 케이스로 전환한다.

    반환: 실제 적용된 설정 dict(로그·보고용). 치환에 실패한 항목이 있으면
    'warnings' 키에 담아 돌려준다(무음 실패 금지).
    """
    p = dict(TRANSIENT_DEFAULTS)
    p.update({k: v for k, v in kw.items() if v is not None})
    warn: List[str] = []

    # ── 1) controlDict ────────────────────────────────────────────────────
    ctrl = case_dir / "system" / "controlDict"
    t = ctrl.read_text()
    for pat, rep, name in [
        (r"application\s+simpleFoam;", "application     pimpleFoam;", "application"),
        (r"^endTimeValue\s+[0-9.eE+-]+;", f"endTimeValue        {p['end_time']:g};", "endTime"),
        (r"^writeIntervalValue\s+[0-9.eE+-]+;",
         f"writeIntervalValue  {p['write_interval']:g};", "writeInterval"),
        (r"deltaT\s+[0-9.eE+-]+;", f"deltaT          {p['delta_t']:g};", "deltaT"),
        (r"writeControl\s+timeStep;", "writeControl    adjustableRunTime;", "writeControl"),
    ]:
        t, ok = _sub_once(t, pat, rep, re.M)
        if not ok:
            warn.append(f"controlDict:{name} 치환 실패")
    # 적응 시간간격 제어는 원본에 없는 항목이라 새로 삽입한다.
    if "adjustTimeStep" not in t:
        # 주석까지 함께 잡아야 삽입한 줄에 원래 주석이 딸려붙지 않는다.
        t, ok = _sub_once(
            t, r"(purgeWrite\s+\d+;[^\n]*)",
            r"\1\n\n"
            f"adjustTimeStep  yes;\nmaxCo           {p['max_co']:g};\n"
            f"maxDeltaT       {p['max_delta_t']:g};")
        if not ok:
            warn.append("controlDict:adjustTimeStep 삽입 실패")
    ctrl.write_text(t)

    # ── 2) fvSchemes — 시간항 ─────────────────────────────────────────────
    sch = case_dir / "system" / "fvSchemes"
    s = sch.read_text()
    # 1줄 포맷과 여러 줄 포맷을 모두 처리
    s, ok = _sub_once(s, r"ddtSchemes\s*\{[^}]*\}",
                      f"ddtSchemes      {{ default {p['ddt_scheme']}; }}", re.S)
    if not ok:
        warn.append("fvSchemes:ddtSchemes 치환 실패")
    sch.write_text(s)

    # ── 3) fvSolution — solvers 확장 + SIMPLE→PIMPLE ──────────────────────
    fvs = case_dir / "system" / "fvSolution"
    f = fvs.read_text()
    # PIMPLE 은 pFinal·UFinal·kFinal·omegaFinal 을 별도로 찾는다. 기존 키를
    # 정규식으로 넓혀 Final 변형까지 같은 설정을 쓰게 한다.
    f, ok1 = _sub_once(f, r"^(\s*)p\s*$", r'\1"p.*"', re.M)
    f, ok2 = _sub_once(f, r'"\(U\|k\|omega\)"', '"(U|k|omega).*"')
    if not (ok1 and ok2):
        warn.append("fvSolution:solvers Final 정규식 확장 실패")
    # SIMPLE 블록 → PIMPLE 블록
    f, ok = _sub_once(
        f, r"SIMPLE\s*\{.*?\n\}",
        ("PIMPLE\n{\n"
         f"    nOuterCorrectors {p['n_outer']};\n"
         f"    nCorrectors {p['n_correctors']};\n"
         f"    nNonOrthogonalCorrectors {p['n_non_orth']};\n"
         "}"), re.S)
    if not ok:
        warn.append("fvSolution:SIMPLE→PIMPLE 치환 실패")
    # nOuterCorrectors=1(=PISO 모드)에서는 완화를 걸지 않는 것이 표준이다.
    f, ok = _sub_once(
        f, r"relaxationFactors\s*\{.*?\n\}",
        ("relaxationFactors\n{\n"
         + ("    equations { \".*\" 1; }\n" if int(p["n_outer"]) <= 1 else
            "    fields    { p 0.3; }\n    equations { \".*\" 0.7; }\n")
         + "}"), re.S)
    if not ok:
        warn.append("fvSolution:relaxationFactors 치환 실패")
    fvs.write_text(f)

    # ── 4) turbulenceProperties — 보완②의 DDES 전환 ───────────────────────
    model = str(p["turbulence"])
    family = TRANSIENT_TURBULENCE_MODELS.get(model, "RAS")
    if family == "LES":
        tp = case_dir / "constant" / "turbulenceProperties"
        tp.write_text(
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            "    class dictionary;\n    object turbulenceProperties;\n}\n\n"
            "simulationType  LES;\n\n"
            "LES\n{\n"
            f"    LESModel        {model};\n"
            "    turbulence      on;\n"
            "    printCoeffs     on;\n"
            "    delta           cubeRootVol;\n"
            "    cubeRootVolCoeffs { deltaCoeff 1; }\n"
            "}\n")
        logger.info(f"[Transient] 난류모델 → LES/{model} (보완②: URANS 비정상성 미포착 대비)")

    # ── 5) force 함수객체 샘플링 간격 ────────────────────────────────────
    # steady 는 writeInterval 10 스텝이면 충분하지만, transient 는 CD(t) 곡선과
    # 통계(평균·RMS)의 해상도가 필요하므로 매 스텝 기록으로 바꾼다.
    t = ctrl.read_text()
    t2, n = re.subn(r"(type\s+force(?:Coeffs)?;(?:[^}]*?))writeInterval\s+\d+;",
                    r"\g<1>writeInterval   1;", t, flags=re.S)
    if n:
        ctrl.write_text(t2)
    else:
        warn.append("controlDict:force 함수객체 writeInterval 조정 실패")

    p["warnings"] = warn
    logger.info(
        f"[Transient] pimpleFoam 전환 완료: endTime={p['end_time']}s "
        f"deltaT={p['delta_t']} maxCo={p['max_co']} "
        f"PIMPLE({p['n_outer']},{p['n_correctors']},{p['n_non_orth']}) model={model}"
        + (f" ⚠️ 경고 {len(warn)}건" if warn else ""))
    for w in warn:
        logger.warning(f"[Transient] {w}")
    return p


def read_force_history(case_dir: Path) -> Dict[str, List[float]]:
    """postProcessing 의 시간이력을 읽어 {time, Cd, Cl, Fx, Fy, Fz, Ftotal} 로 반환.

    forceCoeffs* / coefficient.dat  → Cd, Cl   (v2312 열: Time Cd Cd(f) Cd(r) Cl ...)
    forces*      / force.dat        → Fx,Fy,Fz (Time  total(x y z)  pressure...  viscous...)
    두 파일의 시간축이 다를 수 있으므로 각각 그대로 담고, 통계는 공통 구간에서 낸다.
    """
    out: Dict[str, List[float]] = {k: [] for k in
                                   ("time", "Cd", "Cl", "ftime", "Fx", "Fy", "Fz", "Ftotal")}
    pp = case_dir / "postProcessing"
    if not pp.exists():
        return out

    def _rows(path: Path) -> List[List[float]]:
        rows = []
        for ln in path.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            try:
                rows.append([float(v) for v in
                             ln.replace("(", " ").replace(")", " ").split()])
            except ValueError:
                pass
        return rows

    for d in sorted(pp.glob("forceCoeffs*")):
        for t_dir in sorted(d.iterdir(), key=lambda x: _safe_float(x.name)):
            f = t_dir / "coefficient.dat"
            if not f.exists():
                continue
            for r in _rows(f):
                if len(r) >= 5:
                    out["time"].append(r[0]); out["Cd"].append(r[1]); out["Cl"].append(r[4])

    for d in sorted(pp.glob("forces*")):
        for t_dir in sorted(d.iterdir(), key=lambda x: _safe_float(x.name)):
            f = t_dir / "force.dat"
            if not f.exists():
                continue
            for r in _rows(f):
                if len(r) >= 4:
                    out["ftime"].append(r[0])
                    out["Fx"].append(r[1]); out["Fy"].append(r[2]); out["Fz"].append(r[3])
                    out["Ftotal"].append(math.sqrt(r[1]**2 + r[2]**2 + r[3]**2))
    return out


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


def compute_transient_stats(case_dir: Path,
                            t_avg_start: Optional[float] = None,
                            t_start: float = 0.0) -> Dict[str, Any]:
    """평균구간([t_avg_start, 끝])의 시간평균·RMS·표준편차를 계산한다(지시서 §10).

    t_avg_start 가 None 이면 (t_start + 전체구간의 50%) 를 자동 사용한다.
    반환에는 보완②의 판정용 지표 unsteadiness = std/|mean| 도 포함한다.
    """
    hist = read_force_history(case_dir)
    res: Dict[str, Any] = {"t_avg_start": t_avg_start, "n_samples": 0,
                           "unsteadiness_Cd": None, "is_unsteady": None}
    if not hist["time"] and not hist["ftime"]:
        res["error"] = "시간이력 없음"
        return res

    # simpleFoam 수렴해에서 이어받은 경우, 같은 파일 앞부분에 '반복 횟수' 기반
    # 이력이 남아 있다. 이를 물리시간으로 착각해 평균에 넣으면 수렴 드리프트가
    # 비정상성으로 잘못 집계된다 → transient 시작 시각 이전은 전부 잘라낸다.
    meta = case_dir / "transient_meta.json"
    if meta.exists():
        try:
            t_start = float(json.loads(meta.read_text()).get("t_start", t_start))
        except Exception:
            pass
    if t_start > 0:
        for _tk, _keys in (("time", ("Cd", "Cl")),
                           ("ftime", ("Fx", "Fy", "Fz", "Ftotal"))):
            _keep = [i for i, t in enumerate(hist[_tk]) if t >= t_start]
            hist[_tk] = [hist[_tk][i] for i in _keep]
            for _k in _keys:
                hist[_k] = [hist[_k][i] for i in _keep]
    res["t_start"] = t_start

    if not hist["time"] and not hist["ftime"]:
        res["error"] = "transient 구간 이력 없음"
        return res

    t_end = max(hist["time"] or hist["ftime"])
    if t_avg_start is None:
        t_avg_start = t_start + (t_end - t_start) * 0.5
    elif t_avg_start < t_start:
        # 사용자가 넣은 TavgStart 는 'transient 시작 기준 상대시간'으로 해석한다.
        # (이어받기면 절대시각이 466s 처럼 커서, 5.0 같은 값은 상대값이 자명하다)
        t_avg_start = t_start + t_avg_start
    res["t_avg_start"] = t_avg_start
    res["t_end"] = t_end

    def _stats(times: List[float], vals: List[float], name: str):
        sel = [v for t, v in zip(times, vals) if t >= t_avg_start]
        if not sel:
            return
        n = len(sel)
        mean = sum(sel) / n
        var = sum((v - mean) ** 2 for v in sel) / n
        std = math.sqrt(var)
        res[f"mean_{name}"] = mean
        res[f"std_{name}"] = std
        res[f"rms_{name}"] = math.sqrt(sum(v * v for v in sel) / n)
        res[f"min_{name}"] = min(sel)
        res[f"max_{name}"] = max(sel)
        res["n_samples"] = max(res["n_samples"], n)

    for nm in ("Cd", "Cl"):
        _stats(hist["time"], hist[nm], nm)
    for nm in ("Fx", "Fy", "Fz", "Ftotal"):
        _stats(hist["ftime"], hist[nm], nm)

    # 보완②: 비정상성이 실제로 포착됐는지 판정. 변동이 평균의 1% 미만이면
    # 'URANS 가 비정상성을 억제한 것'으로 보고 DDES 재검증을 권고해야 한다.
    m, s = res.get("mean_Cd"), res.get("std_Cd")
    if m is not None and s is not None and abs(m) > 1e-12:
        res["unsteadiness_Cd"] = s / abs(m)
        res["is_unsteady"] = bool(res["unsteadiness_Cd"] >= 0.01)

    # 항력 변동만으로 판정하면 그물 형상에서 오판한다. 그물은 다수의 가는 실에
    # 힘이 분산돼 각 실의 방출 위상이 상쇄되므로 합력 항력의 변동이 원래 작다.
    # 실측(재설계 목표 75 + DDES): St=0.192 로 와류 방출을 명확히 잡았는데도
    # 변동/평균은 0.41~0.80% 라 종전 기준(≥1%)으로는 '미포착'이 됐다.
    # → 양력의 규칙적 진동(평균선 교차 횟수 + 진폭)을 함께 본다. 이 블록은
    #   is_unsteady 를 True 로 올리기만 하므로 기존 판정이 뒤집히지 않는다.
    _t = [t for t in hist["time"] if t >= t_avg_start]
    _cl = [v for t, v in zip(hist["time"], hist["Cl"]) if t >= t_avg_start]
    if len(_cl) > 10 and len(_t) == len(_cl):
        _mcl = sum(_cl) / len(_cl)
        _cross = sum(1 for i in range(1, len(_cl))
                     if (_cl[i - 1] - _mcl) * (_cl[i] - _mcl) < 0)
        _amp = (max(_cl) - min(_cl)) / 2.0
        res["n_cross_Cl"] = _cross
        res["amp_Cl"] = _amp
        if _cross >= 4 and _t[-1] > _t[0]:
            # 평균선을 한 주기에 두 번 지나므로 주기 = 2 x 구간 / 교차 횟수
            res["period_Cl"] = 2.0 * (_t[-1] - _t[0]) / _cross
            res["freq_Cl"] = 1.0 / res["period_Cl"]
        if m is not None and abs(m) > 1e-12:
            res["unsteadiness_Cl"] = _amp / abs(m)
            if _cross >= 4 and res["unsteadiness_Cl"] >= 0.01:
                res["is_unsteady"] = True
                res["unsteady_by"] = "Cl 진동"
    return res


# ═══════════════════════════════════════════════════════════════════════════
# 단위 셀 모드 케이스 생성기
# ═══════════════════════════════════════════════════════════════════════════

class UnitCellCaseBuilder:
    """
    단위 셀(Unit Cell) 해석 케이스 자동 생성
    - 주기 경계조건(Cyclic) 자동 설정
    - 영각/유속 조합별 케이스 폴더 생성
    - forceCoeffs 함수 자동 설정
    """

    def __init__(self, case_dir: Path, stl_path: Path,
                 speed: float, angle_deg: float,
                 cell_size: float = None,
                 n_cores: Optional[int] = None,
                 nx: int = 1, ny: int = 1,
                 residual_control: float = 1e-4,
                 end_time: int = 2000,
                 write_interval: int = 100,
                 refine_level: int = 3,
                 solidity: float = None,
                 aref_override: Optional[float] = None,
                 strict_mesh_quality: bool = False,
                 rho: float = 1025.0,
                 nu: float = 1.19e-6):
        # 슬리버 셀 억제(비정상 해석의 시간간격 붕괴 방지). 기본 False 라
        # 기존 호출부는 종전과 동일한 격자를 만든다.
        self.strict_mesh_quality = bool(strict_mesh_quality)
        # 유체 물성. 기본값은 템플릿과 같으므로 지정하지 않으면 산출물이 종전과
        # 동일하다(patch_fluid_properties 가 같은 값이면 파일을 건드리지 않는다).
        self.rho              = float(rho)
        self.nu               = float(nu)
        self.case_dir         = case_dir
        self.stl_path         = stl_path
        # 사용자가 UI 에서 직접 지정한 기준면적[m²]. None/0 이하면 자동 계산 사용.
        self.aref_override    = (float(aref_override)
                                 if aref_override and float(aref_override) > 0 else None)
        self.speed            = speed
        self.angle_deg        = angle_deg
        self.nx               = max(1, int(nx))
        self.ny               = max(1, int(ny))
        self.n_cores          = n_cores or get_cpu_count()
        self.residual_control = max(1e-5, min(1e-3, float(residual_control)))
        self.end_time         = max(100, int(end_time))
        self.write_interval   = max(10, int(write_interval))
        # 상한 6: 그물실이 가늘면(망목/실지름 비가 크면) 레벨 5~6 이 필요하다.
        # 종전 상한 4 는 UI 에서 더 올려도 조용히 무시되는 원인이었다.
        self.refine_level     = max(1, min(8, int(refine_level)))

        # cell_size / solidity: 미지정 시 STL에서 자동 감지
        stl_info = detect_stl_cell_size(stl_path)
        if cell_size is None:
            self.cell_size = stl_info["cell_size_mm"] / 1000.0
            logger.info(f"[UnitCell] cell_size 자동 감지: {stl_info['cell_size_mm']:.1f} mm")
        else:
            self.cell_size = float(cell_size)

        if solidity is None:
            self.solidity = stl_info["solidity"]
            logger.info(f"[UnitCell] 고형률 자동 추정: Sn={self.solidity:.4f} "
                        f"(d={stl_info['wire_diameter_mm']:.1f}mm, a={stl_info['cell_size_mm']:.1f}mm)")
        else:
            self.solidity = max(0.01, min(0.95, float(solidity)))

        self.Ux, self.Uy, self.Uz = compute_velocity_vector(speed, angle_deg)
        self.turb       = compute_turbulence_params(speed, length_scale=self.cell_size)

    def build(self) -> Path:
        """케이스 디렉토리 전체 생성 및 반환"""
        logger.info(f"[UnitCell] 케이스 생성 시작: {self.case_dir}")

        # 1. 템플릿 복사
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)
        shutil.copytree(OF_TEMPLATES / "unit_cell", self.case_dir)

        # 2. STL 파일 복사
        trisurf_dir = self.case_dir / "constant" / "triSurface"
        trisurf_dir.mkdir(exist_ok=True)
        shutil.copy2(self.stl_path, trisurf_dir / "netSurface.stl")

        # 3. 경계 조건 파일 수정
        self._patch_velocity_field()
        self._patch_turbulence_fields()
        self._patch_blockMesh()
        self._patch_fvSolution()
        self._patch_controlDict()
        self._patch_snappyLevel()
        self._patch_decomposePar()
        _fp = patch_fluid_properties(self.case_dir, self.rho, self.nu)
        if _fp["changed"]:
            logger.info(f"[UnitCell] 유체 물성 반영: nu={_fp['nu']:.4g} m²/s, "
                        f"rhoInf={_fp['rhoInf']:g} kg/m³ ({', '.join(_fp['changed'])})")

        logger.info(f"[UnitCell] 케이스 생성 완료: {self.case_dir}")
        return self.case_dir

    def _patch_velocity_field(self):
        """속도 경계조건 수정 (영각 적용)"""
        u_file = self.case_dir / "0" / "U"
        Uvec = f"({self.Ux:.6f} {self.Uy:.6f} {self.Uz:.6f})"
        replace_in_file(u_file, {
            "uniform (1.0 0 0);   // 기본값: 1.0 m/s, 0도 영각": f"uniform {Uvec};",
            "uniform (1.0 0 0);": f"uniform {Uvec};",
        })

    def _patch_turbulence_fields(self):
        """k, omega 초기값 수정"""
        k_val = f"{self.turb['k']:.6e}"
        w_val = f"{self.turb['omega']:.6f}"

        for fname, old_val, new_val in [
            ("k",     "3.75e-3", k_val),
            ("omega", "3.45",    w_val),
        ]:
            fpath = self.case_dir / "0" / fname
            text = fpath.read_text()
            text = re.sub(r"uniform\s+[\d.e+-]+;", f"uniform {new_val};", text)
            fpath.write_text(text)

    def _patch_blockMesh(self):
        """blockMeshDict 수정 — 주기(cyclic) 경계조건이므로 격자는 '항상 1셀'만 만든다.
        nx/ny(주기 반복수)는 무한 배열을 대표하는 보고용 값일 뿐, 메시 크기에는
        영향을 주지 않는다(1셀만 풀어도 무한 배열과 동일). 따라서 도메인·격자 수는
        nx/ny와 무관하게 단위 셀 크기 기준으로 고정 → 계산시간이 nx/ny에 불변."""
        half_x = self.cell_size * 1000 / 2   # mm (X 방향 절반, 1셀)
        half_y = self.cell_size * 1000 / 2   # mm (Y 방향 절반, 1셀)
        depth  = self.cell_size * 1000 / 2 * 4         # mm (Z 깊이, 기존 5×→4× 축소)
        # 기준(base) 격자: 셀크기/16 (예 40mm→2.5mm). snappyHexMesh가 그물실 표면
        # 근처를 레벨 2~3으로 세분화하므로 base는 거칠어도 정확도 확보됨.
        # 기존엔 1mm 균일 격자(40×40×200≈32만 셀)로 과도하게 커서 직렬 메싱·해석이
        # 매우 느렸음 → base를 키워 셀 수를 ~15배 줄인다.
        base_mm = max(2.0, self.cell_size * 1000 / 16)
        cells_x = max(8,  round(self.cell_size * 1000 / base_mm))
        cells_y = max(8,  round(self.cell_size * 1000 / base_mm))
        cells_z = max(20, round(2 * depth / base_mm))

        # 단위셀 STL 적합성 점검 — 도메인을 벗어나면 snappy 가 아무것도 못 잡아
        # Cd=0 이 조용히 기록된다. 반드시 눈에 띄게 경고한다.
        try:
            _v = validate_unit_cell_stl(self.stl_path)
            if not _v["ok"]:
                for _iss in _v["issues"]:
                    logger.warning(f"[UnitCell] ⚠️ STL 부적합: {_iss}")
                logger.warning("[UnitCell] ⚠️ 이대로 실행하면 격자가 형상을 잡지 "
                               "못해 Cd=0 이 나올 수 있습니다.")
        except Exception as _e:
            logger.debug(f"[UnitCell] STL 적합성 점검 건너뜀: {_e}")

        # 그물실 격자 해상도 점검 — 망목이 커지면 실 대비 격자가 조용히 거칠어진다.
        # 부족하면 로그로 분명히 알린다(결과를 그대로 믿지 않도록).
        try:
            _wd = detect_stl_cell_size(self.stl_path).get("wire_diameter_mm", 0.0)
            if _wd > 0:
                _tr = twine_resolution(base_mm, _wd, self.refine_level)
                _msg = (f"[UnitCell] 그물실 해상도: 지름 {_wd:.2f} mm / 최소셀 "
                        f"{_tr['finest_mm']:.3f} mm = {_tr['cells_per_d']:.1f} 셀")
                if _tr["ok"]:
                    logger.info(_msg + " ✅")
                else:
                    logger.warning(
                        _msg + f" ⚠️ 목표 {_tr['target']:.0f} 셀 미만 — "
                        f"경계층·박리가 풀리지 않아 Cd 가 부정확할 수 있습니다. "
                        f"정밀화 레벨을 {_tr['required_level']} 이상으로 올리세요.")
        except Exception as _e:
            logger.debug(f"[UnitCell] 해상도 점검 건너뜀: {_e}")

        bmd = self.case_dir / "system" / "blockMeshDict"
        # v11 addLayers: 측면 주기 경계가 cyclicAMI(translational)라
        # separationVector(패치 간 이동벡터 = 셀 크기[m])도 함께 치환한다.
        _sep = self.cell_size   # m (템플릿 자리표시자 0.02 = 20mm 도메인 기준)
        replace_in_file(bmd, {
            "(-10 -10 -50)": f"({-half_x:.1f} {-half_y:.1f} {-depth:.1f})",
            "( 10 -10 -50)": f"({half_x:.1f} {-half_y:.1f} {-depth:.1f})",
            "( 10  10 -50)": f"({half_x:.1f} {half_y:.1f} {-depth:.1f})",
            "(-10  10 -50)": f"({-half_x:.1f} {half_y:.1f} {-depth:.1f})",
            "(-10 -10  50)": f"({-half_x:.1f} {-half_y:.1f} {depth:.1f})",
            "( 10 -10  50)": f"({half_x:.1f} {-half_y:.1f} {depth:.1f})",
            "( 10  10  50)": f"({half_x:.1f} {half_y:.1f} {depth:.1f})",
            "(-10  10  50)": f"({-half_x:.1f} {half_y:.1f} {depth:.1f})",
            "(20 20 100)": f"({cells_x} {cells_y} {cells_z})",
            "separationVector (0.02 0 0);":
                f"separationVector ({_sep:.6f} 0 0);",
            "separationVector (-0.02 0 0);":
                f"separationVector (-{_sep:.6f} 0 0);",
            "separationVector (0 0.02 0);":
                f"separationVector (0 {_sep:.6f} 0);",
            "separationVector (0 -0.02 0);":
                f"separationVector (0 -{_sep:.6f} 0);",
        })

    def _patch_fvSolution(self):
        """수렴 기준(residualControl) 주입"""
        fvs = self.case_dir / "system" / "fvSolution"
        replace_in_file(fvs, {
            "residualLevel   1e-4;": f"residualLevel   {self.residual_control:.0e};",
        })

    def _patch_snappyLevel(self):
        """격자 세분화 레벨 주입.

        [결함 수정] 종전에는 refinementSurfaces 만 레벨을 따라갔고, 피처 에지
        (level 2)와 거리 기반 정밀화 영역(2mm 이내 level 3)이 템플릿에 고정돼
        있었다. 그래서 요청 레벨이 3 이하이면 고정된 영역 레벨 3 이 그 위를
        덮어써, 레벨을 낮춰도 격자가 거의 그대로였다.
            실측: 레벨 2 → 161,514 셀 / 레벨 4 → 165,328 셀 (+2.4%)
        요청 레벨이 템플릿 고정값보다 낮을 때만 함께 낮춘다. 레벨 3 이상에서는
        종전과 완전히 동일한 dict 가 나오므로 기존 결과(lv5~lv8)는 그대로
        유효하다.
        """
        level_min = max(1, self.refine_level - 1)
        level_max = self.refine_level
        # 피처 에지·거리 영역은 '표면보다 촘촘해지지 않도록' 상한을 건다.
        feature_level = min(2, level_min)
        region_level = min(3, level_max)
        snappy = self.case_dir / "system" / "snappyHexMeshDict"
        replace_in_file(snappy, {
            "refineLevelMin  2;": f"refineLevelMin  {level_min};",
            "refineLevelMax  3;": f"refineLevelMax  {level_max};",
            "level   2;":         f"level   {feature_level};",
            "levels  ((2e-3 3));": f"levels  ((2e-3 {region_level}));",
        })
        if self.strict_mesh_quality:
            tighten_mesh_quality(snappy)

    def _patch_controlDict(self):
        """forceCoeffs 기준값 + endTime/writeInterval 주입"""
        ctrl = self.case_dir / "system" / "controlDict"
        import math as _math
        _theta = _math.radians(self.angle_deg)

        # ── Aref(기준면적): 학술 표준 = 고정 기준면적 방식, '1셀' 기준 ──
        # 그물면 법선(=Z축, 정면) 방향으로 투영한 실측 면적을 영각과 무관하게
        # 한 번만 계산한다. 영각 의존성은 Cd(θ)·Cl(θ) 계수에 담기는 것이 정석
        # (Løland 1991; Aarsnes et al. 1990; Kristiansen & Faltinsen 2012 screen
        # model). 원기둥 Cd≈1.2도 기준면적을 d×L로 '고정'하는 것과 동일한 원리.
        #
        # 격자가 항상 1셀(주기 BC)이므로 forceCoeffs의 힘도 1셀분 → Aref도 1셀.
        # nx/ny는 곱하지 않는다. (곱하면 힘=1셀인데 면적만 커져 Cd가 1/(nx·ny)로
        # 잘못 작아짐.) 결과적으로 Cd는 nx/ny에 완전히 불변이다.
        _net_normal = (0.0, 0.0, 1.0)   # 그물면 법선 (Z축 = 와이어 두께 방향)
        try:
            _cell_proj = compute_projected_area(self.stl_path, _net_normal)
        except Exception as _e:
            logger.warning(f"[UnitCell] 투영면적 계산 실패({_e}) → 근사식 사용")
            _cell_proj = 0.0

        if self.aref_override is not None:
            # 사용자 직접 입력이 자동 계산보다 항상 우선한다.
            aref = self.aref_override
            logger.info(
                f"[UnitCell] Aref=사용자 직접 입력 {aref:.6e} m² "
                f"(자동 계산값 {_cell_proj:.6e} m² 무시)")
        elif _cell_proj > 0:
            aref = _cell_proj
            logger.info(
                f"[UnitCell] Aref=고정 기준면적(그물면 법선 투영, 1셀) "
                f"{aref:.6e} m² (영각·nx/ny 무관 — 영각 효과는 Cd/Cl이 표현)")
        else:
            # 폴백: 고형률 × 패널 면적 (2d/a 근사, 1셀)
            aref = self.solidity * (self.cell_size ** 2)
            logger.info(f"[UnitCell] Aref=근사식 Sn×셀² (1셀) = {aref:.6e} m²")

        drag_dir = f"({_math.cos(_theta):.4f} 0 {_math.sin(_theta):.4f})" \
                   if self.speed > 0 else "(1 0 0)"
        # liftDir: 유속에 수직, XZ 평면 내 = (-sinθ, 0, cosθ)
        lift_dir = f"({-_math.sin(_theta):.4f} 0 {_math.cos(_theta):.4f})" \
                   if self.speed > 0 else "(0 0 1)"
        replace_in_file(ctrl, {
            "endTimeValue        2000;":       f"endTimeValue        {self.end_time};",
            "writeIntervalValue  100;":        f"writeIntervalValue  {self.write_interval};",
            "magUInf         1.0;        // 기준 유속 [m/s] - Python에서 교체":
                f"magUInf         {self.speed:.4f};",
            "lRef            0.02;       // 기준 길이 [m] (단위 셀 크기)":
                f"lRef            {self.cell_size:.6f};",
            "Aref            4.0e-4;     // 기준 면적 [m^2] (0.02 x 0.02)":
                f"Aref            {aref:.6e};",
            "liftDir         (0 1 0);    // 양력 방향 (y축)":
                f"liftDir         {lift_dir};",
            "dragDir         (1 0 0);    // 항력 방향 (x축, 유속 방향)":
                f"dragDir         {drag_dir};",
        })

    def _patch_decomposePar(self):
        """CPU 코어 수에 맞게 decomposeParDict 수정"""
        dpd = self.case_dir / "system" / "decomposeParDict"
        replace_in_file(dpd, {
            "numberOfSubdomains  16;": f"numberOfSubdomains  {self.n_cores};",
        })


# ═══════════════════════════════════════════════════════════════════════════
# 전체 구조 모드 케이스 생성기
# ═══════════════════════════════════════════════════════════════════════════

class FullStructureCaseBuilder:
    """
    전체 구조(Full Structure) 해석 케이스 자동 생성
    - 원통형 가두리 STL 자동 처리
    - 도메인 크기 자동 계산
    - 입구/출구/벽면 경계조건 자동 설정
    """

    def __init__(self, case_dir: Path,
                 cage_stl: Optional[Path], net_stl: Optional[Path],
                 speed: float, angle_deg: float = 0.0,
                 cage_diameter: float = 10.0, cage_depth: float = 5.0,
                 n_cores: Optional[int] = None,
                 residual_control: float = 1e-4,
                 end_time: int = 3000,
                 write_interval: int = 100,
                 aref_override: Optional[float] = None,
                 refine_level: int = 3,
                 n_layers: int = 0,
                 auto_refine: bool = False,
                 net_grid_redesign: bool = False,
                 net_grid_target_cells: float = 75.0,
                 net_grid_max_base_cells: int = 3_000_000,
                 wake_box_level: Optional[int] = None,
                 rho: float = 1025.0,
                 nu: float = 1.19e-6):
        # 격자 옵션(보완④: DDES 등에서 격자 민감도를 확인하기 위한 노브).
        # 기본값 refine_level=3 / n_layers=0 은 종전 하드코딩 값과 완전히 동일한
        # snappyHexMeshDict 를 만든다(회귀 방지).
        self.refine_level     = max(1, min(8, int(refine_level)))
        self.n_layers         = max(0, min(10, int(n_layers)))
        self.auto_refine      = bool(auto_refine)
        # 배경격자 재설계(그물처럼 가는 요소가 있는 형상용). 기본 False 라
        # 기존 호출부는 종전과 동일한 격자를 만든다.
        self.net_grid_redesign = bool(net_grid_redesign)
        self.net_grid_target_cells = float(net_grid_target_cells)
        self.net_grid_max_base_cells = int(net_grid_max_base_cells)
        # 후류(정밀화 박스) 레벨 직접 지정. None 이면 종전 자동 규칙 그대로.
        # DES/LES 는 후류가 임계치수당 5셀 이상이어야 LES 모드로 전환되는데,
        # 자동 규칙은 셀 폭발을 막으려 레벨 2 로 묶어둔다. mesh_adequacy() 가
        # 권고 레벨을 내놓아도 종전에는 그것을 적용할 수단이 없었다.
        self.wake_box_level   = (None if wake_box_level is None
                                 else max(0, min(8, int(wake_box_level))))
        # 유체 물성(기본값 = 템플릿 값이므로 미지정 시 산출물 동일)
        self.rho              = float(rho)
        self.nu               = float(nu)
        self.case_dir         = case_dir
        # 사용자가 UI 에서 직접 지정한 기준면적[m²]. None/0 이하면 자동 계산 사용.
        self.aref_override    = (float(aref_override)
                                 if aref_override and float(aref_override) > 0 else None)
        self.cage_stl         = cage_stl
        self.net_stl          = net_stl
        self.speed            = speed
        self.angle_deg        = angle_deg
        self.cage_D           = cage_diameter
        self.cage_H           = cage_depth
        self.n_cores          = n_cores or get_cpu_count()
        self.residual_control = max(1e-5, min(1e-3, float(residual_control)))
        self.end_time         = max(100, int(end_time))
        self.write_interval   = max(10, int(write_interval))
        self.Ux, self.Uy, self.Uz = compute_velocity_vector(speed, angle_deg)
        self.turb          = compute_turbulence_params(speed, length_scale=cage_diameter * 0.1)

    def build(self) -> Path:
        logger.info(f"[FullStructure] 케이스 생성 시작: {self.case_dir}")

        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)
        shutil.copytree(OF_TEMPLATES / "full_structure", self.case_dir)

        # STL 복사
        trisurf_dir = self.case_dir / "constant" / "triSurface"
        trisurf_dir.mkdir(exist_ok=True)
        # 그물(net)만 있고 가두리(cage)가 없으면 'net-only' 모드.
        self._net_only = not (self.cage_stl and self.cage_stl.exists()) \
            and bool(self.net_stl and self.net_stl.exists())
        if self.cage_stl and self.cage_stl.exists():
            shutil.copy2(self.cage_stl, trisurf_dir / "cageSurface.stl")
        if self.net_stl and self.net_stl.exists():
            if self._net_only:
                # 좌표 통일(풍동식): 그물을 Y축으로 α 회전해 저장 → 유속 x 고정과 함께
                # '유속-그물면 상대각 = 90°−α' 가 Unit Cell 과 동일해진다(coordinate_convention).
                self._write_rotated_netSurface(trisurf_dir / "netSurface.stl")
            else:
                shutil.copy2(self.net_stl, trisurf_dir / "netSurface.stl")

        if self._net_only:
            self._compute_net_domain()   # 회전된 netSurface.stl 기준
            self._auto_tune_refine_level()

        self._patch_velocity_fields()
        self._patch_turbulence_fields()
        self._patch_blockMesh()
        self._patch_fvSolution()
        self._patch_snappyHexMesh()
        self._patch_controlDict()
        self._patch_decomposePar()
        _fp = patch_fluid_properties(self.case_dir, self.rho, self.nu)
        if _fp["changed"]:
            logger.info(f"[FullStructure] 유체 물성 반영: nu={_fp['nu']:.4g} m²/s, "
                        f"rhoInf={_fp['rhoInf']:g} kg/m³ ({', '.join(_fp['changed'])})")

        logger.info(f"[FullStructure] 케이스 생성 완료: {self.case_dir}")
        return self.case_dir

    def _write_rotated_netSurface(self, dest: Path):
        """원본 net STL 을 좌표 규약(풍동식)에 따라 Y축으로 α 회전해 ASCII STL 로 저장.
        유속을 x 로 고정하므로, 그물을 회전해 Unit Cell 과 동일한 상대각(90°−α)을 만든다."""
        conv = coordinate_convention(self.angle_deg, "full_structure")
        rot = conv["rotate_geometry_deg"]
        tris = read_stl_triangles(self.net_stl)
        allpts = [v for t in tris for v in t]
        cx = (min(p[0] for p in allpts) + max(p[0] for p in allpts)) / 2.0
        cy = (min(p[1] for p in allpts) + max(p[1] for p in allpts)) / 2.0
        cz = (min(p[2] for p in allpts) + max(p[2] for p in allpts)) / 2.0
        lines = ["solid netSurface"]
        for t in tris:
            rt = rotate_points_y(list(t), rot, (cx, cy, cz))
            # 면 법선(외적) 재계산
            (ax, ay, az), (bx, by, bz), (ccx, ccy, ccz) = rt[0], rt[1], rt[2]
            ux_, uy_, uz_ = bx-ax, by-ay, bz-az
            vx_, vy_, vz_ = ccx-ax, ccy-ay, ccz-az
            nx_, ny_, nz_ = (uy_*vz_-uz_*vy_, uz_*vx_-ux_*vz_, ux_*vy_-uy_*vx_)
            nl = math.sqrt(nx_*nx_+ny_*ny_+nz_*nz_) or 1.0
            lines.append(f"  facet normal {nx_/nl:.6e} {ny_/nl:.6e} {nz_/nl:.6e}")
            lines.append("    outer loop")
            for (vx, vy, vz) in rt:
                lines.append(f"      vertex {vx:.6e} {vy:.6e} {vz:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid netSurface")
        dest.write_text("\n".join(lines) + "\n")

    def _patch_velocity_fields(self):
        if getattr(self, "_net_only", False):
            # 좌표 통일(풍동식): 그물을 회전했으므로 유속은 x 로 고정(항상 정상 유입).
            ux, uy, uz = self.speed, 0.0, 0.0
        else:
            ux, uy, uz = self.Ux, self.Uy, self.Uz
        Uvec = f"({ux:.6f} {uy:.6f} {uz:.6f})"
        u_file = self.case_dir / "0" / "U"
        text = u_file.read_text()
        text = re.sub(r"uniform \(1\.0 0 0\)", f"uniform {Uvec}", text)
        u_file.write_text(text)

    def _patch_turbulence_fields(self):
        k_val = f"{self.turb['k']:.6e}"
        w_val = f"{self.turb['omega']:.6f}"
        for fname, new_val in [("k", k_val), ("omega", w_val)]:
            fpath = self.case_dir / "0" / fname
            text = fpath.read_text()
            text = re.sub(r"uniform\s+[\d.e+-]+;", f"uniform {new_val};", text)
            fpath.write_text(text)

    def _compute_net_domain(self):
        """net STL 의 실제 바운딩박스(scale 0.001 적용 = m)를 읽어 net-only 도메인·
        정밀화 박스·기준점을 계산해 self._dom_* / self._box_* / self._loc 에 저장한다.
        가두리가 없을 때 도메인이 net 크기에 맞아야 snappy 가 net 을 제대로 포착한다."""
        # 회전 적용된 케이스 내 netSurface.stl 을 기준으로(없으면 원본) 도메인 산정.
        _net = self.case_dir / "constant" / "triSurface" / "netSurface.stl"
        tris = read_stl_triangles(_net if _net.exists() else self.net_stl)
        sc = 0.001  # snappyHexMeshDict scale (mm→m)와 일치
        xs = [v[0]*sc for t in tris for v in t]
        ys = [v[1]*sc for t in tris for v in t]
        zs = [v[2]*sc for t in tris for v in t]
        bxmin, bxmax = min(xs), max(xs)
        bymin, bymax = min(ys), max(ys)
        bzmin, bzmax = min(zs), max(zs)
        cx = (bxmin+bxmax)/2; cy = (bymin+bymax)/2; cz = (bzmin+bzmax)/2
        L = max(bxmax-bxmin, bymax-bymin, bzmax-bzmin, 1e-3)
        self._net_L = L; self._net_c = (cx, cy, cz)
        # ── 도메인 ────────────────────────────────────────────────────────
        # 종전: 상류 3L, 하류 7L, 횡·수직 ±3L
        # 재설계(net_grid_redesign): 상류 2L, 하류 5L, 횡·수직 ±2L 로 축소.
        # 배경격자를 '실 지름' 기준으로 잡으면 셀 수가 도메인 부피에 비례해
        # 폭증하므로, 넉넉하던 도메인을 필요 최소로 줄인다(부피 1/2.7).
        if getattr(self, "net_grid_redesign", False):
            _u, _d, _s = 2.0, 5.0, 2.0
        else:
            _u, _d, _s = 3.0, 7.0, 3.0
        self._dom_min = (cx-_u*L, cy-_s*L, cz-_s*L)
        self._dom_max = (cx+_d*L, cy+_s*L, cz+_s*L)
        # 정밀화 박스: net + 근접 후류
        # 정밀화 박스: 체적을 통째로 세분하므로 필요 최소로 잡는다.
        # 후류(유동 +x 하류)는 넉넉히 두되, 상류·측면 여유는 좁힌다.
        # 종전 0.5L 균등 → 상류·측면 0.15L. 하류는 1.5L 유지(후류 해상 필요).
        _m = 0.15 * L
        self._box_min = (bxmin-_m, bymin-_m, bzmin-_m)
        self._box_max = (bxmax+1.5*L, bymax+_m, bzmax+_m)
        # 기준점: net 상류(연결된 유체 영역 어디든 가능, net 표면만 피하면 됨).
        # 도메인 상류 여유가 재설계에서 3L→2L 로 줄었으므로 고정 2.5L 을 쓰면
        # 도메인 밖으로 나간다(snappy FATAL: not inside the mesh).
        # 입구면과 형상 앞면의 중간점으로 잡아 어떤 도메인 설정에도 안전하게 한다.
        # 또한 y=cy, z=cz 를 그대로 쓰면 대칭 도메인에서 정확히 '셀 경계면 위'에
        # 놓여 snappy 가 거부한다(FATAL: not inside the mesh or on a face or edge).
        # 격자선과 겹치지 않도록 L 의 소수 비율만큼 어긋나게 둔다.
        self._loc = ((self._dom_min[0] + bxmin) / 2.0,
                     cy + 0.0137 * L, cz + 0.0219 * L)

    def _auto_tune_refine_level(self):
        """형상의 '가장 가는 치수'를 기준으로 정밀화 레벨을 자동 산정한다.

        배경격자는 형상 전체 크기(L/8)로 정해지므로, 그물처럼 큰 영역에 가는
        요소가 흩어진 형상은 기본 레벨 3 에서 실 지름당 1~2 셀밖에 안 걸린다.
        (3by3 패널 실측: 레벨 3 → 1.5 셀, Cd 가 레벨 6 대비 43% 과대평가)
        STL 종류와 무관하게 '최소 두께 / 최소셀' 이 목표치를 넘도록 레벨을 올린다.

        auto_refine=False 면 진단만 로그로 남기고 레벨은 바꾸지 않는다.
        """
        try:
            _thin = self._thin_dimension_mm()
            if _thin <= 0:
                return
            _base_mm = self._base_cell_mm()
            _tr = twine_resolution(_base_mm, _thin, self.refine_level)
            _msg = (f"[FullStructure] 형상 최소두께 {_thin:.2f} mm / 최소셀 "
                    f"{_tr['finest_mm']:.3f} mm = {_tr['cells_per_d']:.1f} 셀")
            if _tr["ok"]:
                logger.info(_msg + " ✅")
                return
            if not self.auto_refine:
                logger.warning(
                    _msg + f" ⚠️ 목표 {_tr['target']:.0f} 셀 미만 — 정밀화 레벨을 "
                    f"{_tr['required_level']} 이상으로 올려야 합니다. "
                    f"(자동 보정이 꺼져 있어 레벨 {self.refine_level} 로 진행)")
                return
            _new = min(8, _tr["required_level"])
            if _new > self.refine_level:
                logger.warning(
                    _msg + f" ⚠️ 목표 미만 → 정밀화 레벨 자동 상향 "
                    f"{self.refine_level} → {_new} (셀 수·계산시간 증가)")
                self.refine_level = _new
                if _tr["required_level"] > 8:
                    logger.warning(
                        f"[FullStructure] 목표 달성에는 레벨 {_tr['required_level']} 이 "
                        f"필요하지만 상한 8 로 제한했습니다. 결과에 격자 오차가 "
                        f"남습니다.")
        except Exception as _e:
            logger.debug(f"[FullStructure] 정밀화 자동 산정 건너뜀: {_e}")

    def _base_cell_mm(self) -> float:
        """_patch_blockMesh 와 동일한 배경격자 크기[mm] 산정(자동 산정용)."""
        L = self._net_L
        x0, y0, z0 = self._dom_min
        x1, y1, z1 = self._dom_max
        _cell = max(L / 8.0, 1e-4)
        nx = max(40, min(120, int((x1 - x0) / _cell)))
        return (x1 - x0) / max(nx, 1) * 1000.0

    def _thin_dimension_mm(self) -> float:
        """형상에서 격자가 반드시 해상해야 할 '가장 가는 치수'[mm].

        그물이면 그물실 지름, 카이트·판재면 두께에 해당한다. 회전 전 원본 STL 의
        바운딩박스 최소변을 쓴다(STL 종류와 무관하게 동작).
        """
        tris = read_stl_triangles(self.net_stl)
        if not tris:
            return 0.0
        pts = [v for t in tris for v in t]
        spans = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)]
        return min(s for s in spans if s > 0) if any(s > 0 for s in spans) else 0.0

    def _patch_blockMesh(self):
        """도메인 크기 자동 계산. net-only 면 net 크기 기준, 아니면 가두리 크기 기준."""
        if getattr(self, "_net_only", False):
            x_min, y_min, z_min = self._dom_min
            x_max, y_max, z_max = self._dom_max
            L = self._net_L
            if getattr(self, "net_grid_redesign", False):
                # ── 배경격자 재설계 ──────────────────────────────────────
                # 종전 L/8 은 '형상 전체 크기' 기준이라, 그물처럼 가는 요소가
                # 흩어진 형상은 실 지름당 셀이 1~2 개밖에 안 걸린다(3by3 실측
                # 레벨 3 → 1.5셀). 문헌값에 수렴하려면 70셀 이상이 필요한데,
                # L/8 기준으로는 레벨 9~10 이 필요해 실행 불가였다.
                #
                # 재설계: '임계 치수(실 지름)'를 기준으로 base 를 잡는다.
                #   base = crit × 2^level / TARGET
                # 이러면 지정한 레벨에서 곧바로 TARGET 셀/지름이 나온다.
                # 배경 셀 총수가 폭증하지 않도록 상한을 두고, 넘으면 base 를
                # 키운다(그만큼 해상도가 떨어지므로 로그로 알린다).
                _crit_mm = self._thin_dimension_mm() or (L * 1000 / 8)
                _target = float(getattr(self, "net_grid_target_cells", 75.0))
                _lvl = int(self.refine_level)
                # 산식은 net_grid_base_cell() 로 분리했다(UI 격자 적정성 판정이
                # 같은 값을 쓰기 위해서다). 아래 로그는 종전 그대로 남긴다.
                _cell = (_crit_mm / 1000.0) * (2 ** _lvl) / _target
                # [필수 제약] 배경 셀이 임계 치수보다 크면 snappyHexMesh 가 표면을
                # 애초에 찾지 못해 정밀화가 시작조차 안 된다(실측: 배경 9.6mm /
                # 실 3mm 에서 레벨 2에 중단 → 실 지름당 1.25셀). 배경 셀을 임계
                # 치수 이하로 강제한다. (실측: 배경/임계 = 1.6배는 정상 작동,
                #  3.2배는 실패 → 1.0배를 안전 기준으로 둔다.)
                _cap_cell = (_crit_mm / 1000.0)
                if _cell > _cap_cell:
                    logger.info(
                        f"[FullStructure] 배경셀 {_cell*1000:.3f}mm 가 임계치수 "
                        f"{_crit_mm:.2f}mm 의 1/2 를 초과 → {_cap_cell*1000:.3f}mm 로 제한 "
                        f"(그렇지 않으면 표면 정밀화가 시작되지 않음)")
                    _cell = _cap_cell
                _cap = int(getattr(self, "net_grid_max_base_cells", 3_000_000))
                _cell = net_grid_base_cell(
                    _crit_mm, _lvl, _target,
                    domain_m=(x_max-x_min, y_max-y_min, z_max-z_min),
                    max_base_cells=_cap)
                _got = _crit_mm / (_cell * 1000 / (2 ** _lvl))
                logger.info(
                    f"[FullStructure] 배경격자 재설계: 임계치수 {_crit_mm:.2f}mm, "
                    f"레벨 {_lvl} → 배경셀 {_cell*1000:.3f}mm, "
                    f"최소셀 {_cell*1000/(2**_lvl):.4f}mm = {_got:.1f} 셀/지름 "
                    f"(목표 {_target:.0f})")
                nx = max(20, int((x_max-x_min)/_cell))
                ny = max(20, int((y_max-y_min)/_cell))
                nz = max(20, int((z_max-z_min)/_cell))
            else:
                # 기저 셀 ~ L/8, 도메인 비율에 맞춰 분할 수 산정(40~120 클램프)
                _cell = max(L/8.0, 1e-4)
                nx = max(40, min(120, int((x_max-x_min)/_cell)))
                ny = max(30, min(100, int((y_max-y_min)/_cell)))
                nz = max(30, min(100, int((z_max-z_min)/_cell)))
        else:
            D, H = self.cage_D, self.cage_H
            # 도메인: 상류 3D, 하류 7D, 횡방향 3D, 수심 H
            x_min = -3 * D;  x_max = 7 * D
            y_min = -3 * D;  y_max = 3 * D
            z_min = -H;      z_max = 0.0
            nx = max(50, int(10 * D))
            ny = max(40, int( 6 * D))
            nz = max(10, int( 4 * H))

        bmd = self.case_dir / "system" / "blockMeshDict"
        replace_in_file(bmd, {
            "(-30  -30  -5)": f"({x_min:.6g}  {y_min:.6g}  {z_min:.6g})",
            "( 70  -30  -5)": f"({x_max:.6g}  {y_min:.6g}  {z_min:.6g})",
            "( 70   30  -5)": f"({x_max:.6g}  {y_max:.6g}  {z_min:.6g})",
            "(-30   30  -5)": f"({x_min:.6g}  {y_max:.6g}  {z_min:.6g})",
            "(-30  -30   0)": f"({x_min:.6g}  {y_min:.6g}  {z_max:.6g})",
            "( 70  -30   0)": f"({x_max:.6g}  {y_min:.6g}  {z_max:.6g})",
            "( 70   30   0)": f"({x_max:.6g}  {y_max:.6g}  {z_max:.6g})",
            "(-30   30   0)": f"({x_min:.6g}  {y_max:.6g}  {z_max:.6g})",
            "(100 60 20)": f"({nx} {ny} {nz})",
        })

    def _patch_snappyHexMesh(self):
        """가두리 크기에 맞게 정밀화 박스 조정. net-only 면 cageSurface 참조가 없는
        snappyHexMeshDict 를 새로 써서 net 만으로 메싱한다."""
        snappy = self.case_dir / "system" / "snappyHexMeshDict"
        if getattr(self, "_net_only", False):
            self._write_netonly_snappy(snappy)
            return
        D, H = self.cage_D, self.cage_H
        r = D / 2 * 1.2
        replace_in_file(snappy, {
            "min     (-6 -6 -6);": f"min     ({-r:.2f} {-r:.2f} {-(H+1):.2f});",
            "max     ( 6  6  1);": f"max     ({r:.2f}  {r:.2f}  1.0);",
            "locationInMesh (0 0 -2.5);": f"locationInMesh (0 0 {-H/2:.2f});",
        })

    def _write_netonly_snappy(self, snappy_path):
        """그물만 있는 경우의 snappyHexMeshDict 생성 — cageSurface 참조 없이
        netSurface 만 정밀화. net STL 은 열린 면(시트)이라 inside 제거 없이 표면 주변만
        정밀화한다(unit_cell 의 net 처리와 동일 사상). 박스·기준점은 net 바운드 기반."""
        bx0, by0, bz0 = self._box_min
        bx1, by1, bz1 = self._box_max
        lx, ly, lz = self._loc
        # refine_level=3 → (2 3) 로 종전과 동일. n_layers=0 → addLayers false.
        _lmax = int(self.refine_level)
        _lmin = max(1, _lmax - 1)
        # 정밀화 박스(refineBox)는 '체적'을 통째로 세분하므로 표면 레벨을 그대로
        # 따라가면 셀이 폭발한다. 그물처럼 가는 형상은 표면만 깊게 파고 박스는
        # 얕게 둬야 한다(레벨 3 까지는 종전대로 _lmin 을 써서 회귀 없음).
        _lbox = _lmin if _lmax <= 3 else min(_lmin, 2)
        # 사용자가 후류 레벨을 직접 지정하면 그 값을 쓴다(DES/LES 후류 요건).
        if self.wake_box_level is not None:
            _lbox = min(_lmax, self.wake_box_level)
        _add  = "true" if self.n_layers > 0 else "false"
        # n_layers=0 이면 종전 출력('layers {}')과 바이트까지 동일해야 하므로
        # 공백을 넣지 않는다.
        _lay  = (f" netSurface {{ nSurfaceLayers {self.n_layers}; }} "
                 if self.n_layers > 0 else "")
        content = f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       {_add};

geometry
{{
    netSurface.stl
    {{
        type    triSurfaceMesh;
        name    netSurface;
        scale   0.001;
    }}
    refineBox
    {{
        type    searchableBox;
        min     ({bx0:.5f} {by0:.5f} {bz0:.5f});
        max     ({bx1:.5f} {by1:.5f} {bz1:.5f});
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      8000000;
    minRefinementCells  10;
    maxLoadUnbalance    -1;
    nCellsBetweenLevels 3;

    features ( {{ file "netSurface.eMesh"; level {_lmin}; }} );

    refinementSurfaces
    {{
        netSurface
        {{
            level ({_lmin} {_lmax});
            patchInfo {{ type wall; inGroups (wall); }}
        }}
    }}

    refinementRegions
    {{
        refineBox {{ mode inside; levels ((1e10 {_lbox})); }}
    }}

    resolveFeatureAngle 30;
    locationInMesh ({lx:.5f} {ly:.5f} {lz:.5f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{{_lay}}}
    expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0; nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1;
    minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}

writeFlags ( scalarLevels );
mergeTolerance 1e-6;
"""
        snappy_path.write_text(content)

    def _patch_fvSolution(self):
        """수렴 기준(residualControl) 주입"""
        fvs = self.case_dir / "system" / "fvSolution"
        replace_in_file(fvs, {
            "residualLevel   1e-4;": f"residualLevel   {self.residual_control:.0e};",
        })

    def _patch_controlDict(self):
        ctrl = self.case_dir / "system" / "controlDict"
        if getattr(self, "_net_only", False):
            # net-only: 힘은 netSurface 패치에서만 계산. 기준면적은 그물면(법선 z)
            # 투영 실측, 기준길이는 net 특성치 L. magUInf=유속.
            try:
                aref = compute_projected_area(self.net_stl, (0.0, 0.0, 1.0))
            except Exception:
                aref = 0.0
            if not (aref and aref > 1e-9):
                aref = max(self._net_L**2, 1e-6)
            if self.aref_override is not None:
                # 사용자 직접 입력이 자동 계산보다 항상 우선한다.
                logger.info(f"[FullStructure] Aref=사용자 직접 입력 "
                            f"{self.aref_override:.6e} m² (자동 {aref:.6e} m² 무시)")
                aref = self.aref_override
            else:
                logger.info(f"[FullStructure] Aref=자동(그물면 법선 투영) {aref:.6e} m²")
            lref = self._net_L
            cx, cy, cz = self._net_c
            # ── CofR(모멘트 기준점) = 형상 중심 ──────────────────────────────
            # net-only 는 STL 좌표를 그대로 쓰므로 형상이 원점에서 멀리 떨어져 있을
            # 수 있다(예: 카이트가 x≈0.78 m). 템플릿 기본값 (0 0 0) 을 두면 Cm 이
            # '원점까지의 지렛대 × 힘'에 지배되어 공력 피칭모멘트가 아니게 된다.
            # 형상 중심으로 옮겨야 Cm 이 의미를 갖는다. (Cd·Cl 은 CofR 무관.)
            # 치환은 forceCoeffs·forces 두 함수객체의 CofR 을 함께 갱신한다.
            replace_in_file(ctrl, {
                "endTimeValue        3000;": f"endTimeValue        {self.end_time};",
                "writeIntervalValue  100;":  f"writeIntervalValue  {self.write_interval};",
                "magUInf         1.0;": f"magUInf         {self.speed:.4f};",
                "lRef            10.0;":  f"lRef            {lref:.6f};",
                "Aref            50.0;":  f"Aref            {aref:.6e};",
                "CofR            (0 0 0);":
                    f"CofR            ({cx:.6f} {cy:.6f} {cz:.6f});",
                "patches         (cageSurface netSurface);": "patches         (netSurface);",
            })
            logger.info(f"[FullStructure] CofR=형상 중심 "
                        f"({cx:.4f} {cy:.4f} {cz:.4f}) m — Cm 기준점")
            # 샘플링 라인을 net 중심 부근으로(도메인 밖이면 무의미하므로)
            replace_in_file(ctrl, {
                "start       (-30 0 -2.5);": f"start       ({cx-2*self._net_L:.4f} {cy:.4f} {cz:.4f});",
                "end         ( 70 0 -2.5);": f"end         ({cx+5*self._net_L:.4f} {cy:.4f} {cz:.4f});",
                "start       (15 -15 -2.5);": f"start       ({cx:.4f} {cy-2*self._net_L:.4f} {cz:.4f});",
                "end         (15  15 -2.5);": f"end         ({cx:.4f} {cy+2*self._net_L:.4f} {cz:.4f});",
                "start       (35 -15 -2.5);": f"start       ({cx+self._net_L:.4f} {cy-2*self._net_L:.4f} {cz:.4f});",
                "end         (35  15 -2.5);": f"end         ({cx+self._net_L:.4f} {cy+2*self._net_L:.4f} {cz:.4f});",
            })
            return
        D, H = self.cage_D, self.cage_H
        aref = D * H
        if self.aref_override is not None:
            logger.info(f"[FullStructure] Aref=사용자 직접 입력 "
                        f"{self.aref_override:.6e} m² (자동 D×H {aref:.4f} m² 무시)")
            aref = self.aref_override
        replace_in_file(ctrl, {
            "endTimeValue        3000;":       f"endTimeValue        {self.end_time};",
            "writeIntervalValue  100;":        f"writeIntervalValue  {self.write_interval};",
            "magUInf         1.0;": f"magUInf         {self.speed:.4f};",
            "lRef            10.0;":  f"lRef            {D:.4f};",
            "Aref            50.0;":  f"Aref            {aref:.6e};",
        })
        # 샘플링 라인 좌표 교체
        replace_in_file(ctrl, {
            "start       (-30 0 -2.5);": f"start       ({-3*D:.1f} 0 {-H/2:.2f});",
            "end         ( 70 0 -2.5);": f"end         ({7*D:.1f} 0 {-H/2:.2f});",
            "start       (15 -15 -2.5);": f"start       ({D*1.5:.1f} {-D*1.5:.1f} {-H/2:.2f});",
            "end         (15  15 -2.5);": f"end         ({D*1.5:.1f} {D*1.5:.1f} {-H/2:.2f});",
            "start       (35 -15 -2.5);": f"start       ({D*3.5:.1f} {-D*1.5:.1f} {-H/2:.2f});",
            "end         (35  15 -2.5);": f"end         ({D*3.5:.1f} {D*1.5:.1f} {-H/2:.2f});",
        })

    def _patch_decomposePar(self):
        dpd = self.case_dir / "system" / "decomposeParDict"
        replace_in_file(dpd, {
            "numberOfSubdomains  16;": f"numberOfSubdomains  {self.n_cores};",
        })


# ═══════════════════════════════════════════════════════════════════════════
# OpenFOAM 실행 관리자
# ═══════════════════════════════════════════════════════════════════════════

class OpenFOAMRunner:
    """
    OpenFOAM 해석 실행 및 모니터링
    - MPI 병렬 자동 실행
    - 실시간 로그 스트리밍
    - 잔차(residual) 파싱 및 수렴 판정
    - 진행률 콜백 제공
    """

    def __init__(self, case_dir: Path, n_cores: int = 8,
                 of_version: str = "v2312",
                 progress_cb: Optional[Callable] = None,
                 log_cb: Optional[Callable] = None,
                 step_cb: Optional[Callable] = None):
        self.case_dir    = case_dir
        self.n_cores     = n_cores
        self.of_version  = of_version
        self.progress_cb = progress_cb   # progress_cb(percent, step, max_step)
        self.log_cb      = log_cb         # log_cb(line: str)
        self.step_cb     = step_cb        # step_cb(module, status, pct, detail)
        self.log_file    = LOGS_DIR / f"{case_dir.name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        self._proc       = None
        self._stop_flag  = threading.Event()
        self.converged   = False
        self.last_residuals: Dict[str, float] = {}

    # ─── OpenFOAM 소스 환경 감지 ──────────────────────────────────────────
    @staticmethod
    def find_openfoam_bashrc() -> Optional[str]:
        candidates = [
            "/opt/openfoam10/etc/bashrc",
            "/opt/openfoam9/etc/bashrc",
            "/opt/openfoam8/etc/bashrc",
            "/usr/lib/openfoam/openfoam2312/etc/bashrc",
            "/usr/lib/openfoam/openfoam2206/etc/bashrc",
            os.path.expanduser("~/OpenFOAM/OpenFOAM-v2312/etc/bashrc"),
            os.path.expanduser("~/OpenFOAM/OpenFOAM-v2206/etc/bashrc"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def _of_cmd(self, cmd: str) -> str:
        bashrc = self.find_openfoam_bashrc()
        if bashrc:
            # bash -c로 명시적 실행 (shell=True 환경에서도 확실히 소스 적용)
            return f'bash -c "source {bashrc} && {cmd}"'
        return cmd  # 이미 PATH에 포함된 경우

    # ─── 단계별 실행 메서드 ────────────────────────────────────────────────

    def run_blockMesh(self) -> bool:
        return self._run_step("blockMesh", "배경 격자 생성 (blockMesh)")

    def run_surfaceFeatureExtract(self) -> bool:
        # surfaceFeatureExtractDict 생성
        self._write_surfaceFeatureExtractDict()
        return self._run_step("surfaceFeatureExtract", "표면 피처 추출")

    def _has_cyclic_patches(self) -> bool:
        """blockMesh 생성 후 boundary 파일에서 cyclic 패치 존재 여부 확인"""
        boundary = self.case_dir / "constant" / "polyMesh" / "boundary"
        if boundary.exists():
            return "cyclic" in boundary.read_text()
        return False

    def run_snappyHexMesh(self) -> bool:
        # 항상 직렬 실행. 병렬 snappy 는 decomposePar(블록메시 — netSurface 등 snappy 가
        # 새로 만드는 wall 패치가 아직 없음) 이후 실행되는데, decomposePar 는 메시에
        # 없는 패치의 필드 boundaryField 항목을 떨어뜨린다. 그 뒤 병렬 snappy 가
        # netSurface 패치를 추가해도 processor 필드엔 항목이 없어 simpleFoam 이
        # 'Cannot find patchField entry for netSurface' 로 즉시 실패한다(full_structure
        # 가 늘 실패하던 원인). 직렬로 메싱해 패치를 먼저 생성한 뒤 run_solver 에서
        # decomposePar 로 분할하면 필드 항목이 보존된다. (cyclic 도 동일하게 직렬.)
        return self._run_step(
            "snappyHexMesh -overwrite", "격자 스냅 (snappyHexMesh, 직렬)")

    def run_decomposePar(self) -> bool:
        return self._run_step("decomposePar -force", "도메인 분할 (decomposePar)")

    def run_solver(self, end_time: int = 2000, solver: str = "simpleFoam") -> bool:
        """병렬 솔버 실행 + 실시간 잔차 모니터링.

        solver 기본값은 simpleFoam 이라 기존 호출부(인자 없이 호출)는 종전과
        완전히 동일하게 동작한다. transient 는 solver="pimpleFoam" 으로 전달한다.
        end_time 은 simpleFoam 이면 반복 횟수, pimpleFoam 이면 물리시간[s] 이며
        진행률 계산은 양쪽 모두 step/end_time 비율로 동일하게 처리된다.
        """
        solver = solver if solver in ("simpleFoam", "pimpleFoam") else "simpleFoam"
        cmd = f"mpirun --oversubscribe -np {self.n_cores} {solver} -parallel"
        # snappyHexMesh 를 직렬로 돌리므로(위 참조) 여기서 분할한다. 직렬 snappy 가
        # 이미 netSurface 등 패치를 만든 메시를 기준으로 decomposePar 하므로 processor
        # 필드에 패치 항목이 보존된다.
        pre = "decomposePar -force"
        return self._run_step(cmd, f"CFD 해석 ({solver})", parallel=True,
                              pre_cmd=pre,
                              monitor_residuals=True, end_time=end_time)

    def run_reconstructPar(self) -> bool:
        return self._run_step("reconstructPar -latestTime", "결과 재조합 (reconstructPar)")

    # ─── 비정상(transient) 실행 ───────────────────────────────────────────

    def _perturb_initial_field(self, magnitude: float = 0.01) -> None:
        """0/U 의 내부장에 미소 비대칭 성분을 넣어 대칭을 깬다(보완③).

        구·원기둥처럼 완전 대칭인 형상은 대칭 초기장에서 출발하면 대칭을 깨뜨릴
        요인이 없어 와류 방출이 시작되지 않을 수 있다. 자유류의 magnitude 배(기본
        1%)만큼 횡방향 성분을 더해 초기 대칭만 깬다. 경계조건은 건드리지 않는다
        (지시서 §2: BC 생성 로직 불변).
        """
        u_file = self.case_dir / "0" / "U"
        if not u_file.exists():
            self._emit_log("⚠️ 교란 주입 건너뜀 — 0/U 없음")
            return
        txt = u_file.read_text()
        m = re.search(r"internalField\s+uniform\s+\(([^)]*)\)", txt)
        if not m:
            self._emit_log("⚠️ 교란 주입 건너뜀 — internalField 패턴 불일치")
            return
        try:
            ux, uy, uz = [float(v) for v in m.group(1).split()]
        except ValueError:
            self._emit_log("⚠️ 교란 주입 건너뜀 — 속도 파싱 실패")
            return
        mag = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        d = mag * float(magnitude)
        new = f"internalField   uniform ({ux:.6f} {uy + d:.6f} {uz + d * 0.5:.6f})"
        u_file.write_text(txt[:m.start()] + new + txt[m.end():])
        self._emit_log(f"🌀 대칭 교란 주입: 자유류의 {magnitude*100:.1f}% "
                       f"(Uy +{d:.4f}, Uz +{d*0.5:.4f} m/s)")

    def _run_transient(self, duration: float, cfg: Dict[str, Any]) -> bool:
        """pimpleFoam 실행. 필요 시 simpleFoam 선행 수렴을 초기조건으로 쓴다."""
        init = bool(cfg.get("init_from_steady", True))
        t0 = 0.0

        if init:
            steady_iters = int(cfg.get("steady_end_time", 1000))
            self._emit_log(f"1단계: simpleFoam 선행 수렴 ({steady_iters}회) — "
                           "pimpleFoam 초기조건 생성")
            if not self.run_solver(steady_iters, solver="simpleFoam"):
                return False
            if self._stop_flag.is_set():
                return False
            # 병렬 결과는 processor*/ 에 남아 있다. 재분할하면 이 결과가 날아가므로
            # 그대로 이어받되, '시간'을 0 으로 되돌린다.
            #
            # [중요] simpleFoam 은 반복 횟수를 시간으로 쓰므로 선행 수렴 후 t=400
            # 같은 값이 된다. 여기서 물리 시간간격(예 1e-5 s)으로 이어가면
            # 400.00001 을 표현해야 하는데 timePrecision 6 으로는 불가능해
            # 시간이 전진하지 못하고 'Starting time loop → End' 로 즉시 끝난다.
            # → 수렴장을 0 시간 디렉토리로 옮겨 물리시간 0 부터 다시 시작한다.
            _latest = self._latest_processor_time()
            if _latest > 0:
                self._reset_processor_time_to_zero(_latest)
            t0 = 0.0
            self._emit_log(f"선행 수렴 완료(t={_latest:g}) — 수렴장을 t=0 으로 옮겨 "
                           f"비정상 해석 시작")
        else:
            if bool(cfg.get("perturb", False)):
                self._perturb_initial_field(float(cfg.get("perturb_magnitude", 0.01)))

        # 케이스를 transient 로 전환
        end_abs = t0 + float(duration)
        tcfg = {k: v for k, v in cfg.items()
                if k in ("delta_t", "max_co", "max_delta_t", "write_interval",
                         "n_outer", "n_correctors", "n_non_orth", "turbulence",
                         "ddt_scheme")}
        applied = apply_transient_settings(self.case_dir, end_time=end_abs, **tcfg)
        if applied.get("warnings"):
            for w in applied["warnings"]:
                self._emit_log(f"⚠️ transient 설정: {w}")
        # 이어받기든 아니든 t=0 에서 시작한다(위에서 수렴장을 0 으로 옮겼다).
        ctrl = self.case_dir / "system" / "controlDict"
        c = ctrl.read_text()
        c = re.sub(r"startFrom\s+\w+;", "startFrom       startTime;", c)
        c = re.sub(r"startTime\s+[0-9.eE+-]+;", "startTime       0;", c)
        ctrl.write_text(c)
        self._transient_start_time = t0
        # 이어받기면 forceCoeffs/forces 파일에 simpleFoam 반복 이력(t=0..t0)과
        # pimpleFoam 물리시간 이력(t0..)이 같이 쌓인다. 통계에서 앞부분을 반드시
        # 잘라내야 하므로 시작 시각을 케이스에 남긴다.
        try:
            (self.case_dir / "transient_meta.json").write_text(json.dumps(
                {"t_start": t0, "solver": "pimpleFoam",
                 "init_from_steady": init, "end_time": end_abs},
                ensure_ascii=False, indent=2))
        except Exception as _e:
            self._emit_log(f"⚠️ transient_meta.json 저장 실패: {_e}")

        # ── 사전 진단(프로브) — 본 실행 전에 실제 deltaT 를 재본다 ──────────
        # 이게 없으면 시간간격이 붕괴한 케이스에서 수 시간을 쓰고 결과 없이 끝난다.
        _max_steps = int(cfg.get("max_steps", 200000))
        _pr = self.probe_transient_timestep(
            float(applied.get("delta_t", 1e-3)), float(applied.get("max_co", 0.8)),
            float(duration))
        if _pr["ok"]:
            # 스텝당 소요시간은 프로브가 실측한다. 실측에 실패한 경우에만 0.4초를
            # 가정하고, 가정임을 로그에 명시한다(종전에는 항상 가정값이었다).
            _spc = _pr.get("sec_per_step")
            _h = _pr["steps"] * (_spc if _spc else 0.4) / 3600.0
            _how = (f"실측 {_spc:.2f}초/스텝" if _spc else "스텝당 0.4초 가정")
            self._emit_log(
                f"🔎 사전 진단: Co_max={_pr['co_max']:.3g} → 실제 deltaT "
                f"{_pr['delta_t']:.3g}s → 필요 스텝 {_pr['steps']:,.0f}개 "
                f"(대략 {_h:.1f}시간, {_how})")
            if _pr["steps"] > _max_steps:
                self._emit_log(
                    f"❌ 실행 중단 — 필요 스텝 {_pr['steps']:,.0f}개가 한계 "
                    f"{_max_steps:,}개를 초과합니다.\n"
                    f"   원인: 격자에 매우 얇은 셀이 있거나 초기장이 불안정해 "
                    f"Courant 수가 {_pr['co_max']:.3g} 까지 올라갑니다.\n"
                    f"   해결 방법:\n"
                    f"   · 정밀화 레벨을 한 단계 낮추기 (가장 확실, 정확도 일부 손실)\n"
                    f"   · 격자 품질 기준 강화 옵션(strict_mesh_quality) 켜기\n"
                    f"   · 물리시간을 {duration*_max_steps/_pr['steps']:.4g}s 이하로 "
                    f"줄여 경향만 확인\n"
                    f"   · nOuterCorrectors 를 올리고 maxCo 를 높여 Co>1 허용"
                    f"(이 경우 스텝당 비용이 배수로 증가)")
                return False
        else:
            self._emit_log(f"⚠️ 사전 진단 건너뜀 ({_pr['reason']}) — 그대로 진행합니다")

        self._emit_log(f"2단계: pimpleFoam 비정상 해석 (t={t0:g} → {end_abs:g} s)")
        # 이어받기면 이미 분할된 processor 결과를 써야 하므로 재분할 금지
        cmd = f"mpirun --oversubscribe -np {self.n_cores} pimpleFoam -parallel"
        return self._run_step(cmd, "CFD 해석 (pimpleFoam)", parallel=True,
                              pre_cmd=None if init else "decomposePar -force",
                              monitor_residuals=True, end_time=end_abs,
                              start_time=t0)

    def probe_transient_timestep(self, delta_t: float, max_co: float,
                                 duration: float) -> Dict[str, Any]:
        """본 실행 전에 실제 가능한 deltaT 를 측정한다(수 초 소요).

        pimpleFoam 은 시간 루프에 들어가기 '전에' 초기 Courant 수를 출력한다.
        endTime 을 startTime 과 같게 두면 루프를 돌지 않고 그 값만 찍고 끝나므로,
        수 초 만에 다음을 알 수 있다.

            deltaT_실제 = deltaT_설정 × maxCo / Co_max
            필요 스텝   = duration / deltaT_실제

        격자 슬리버든 초기장 이상이든 원인과 무관하게 잡히는 것이 장점이다.
        (실측 사례: 레벨 6 단위셀에서 Co_max=297.5 → deltaT 2.7e-8 → 743만 스텝.
         이 진단이 없어 14.7분을 쓰고 결과 없이 끝났다.)
        """
        out: Dict[str, Any] = {"ok": False, "co_max": None, "delta_t": None,
                               "steps": None, "reason": ""}
        ctrl = self.case_dir / "system" / "controlDict"
        if not ctrl.exists():
            out["reason"] = "controlDict 없음"
            return out
        backup = ctrl.read_text()
        try:
            # endTime 을 startTime(=0) 과 같게 → 시간 루프 진입 즉시 종료
            probe = re.sub(r"^endTimeValue\s+[0-9.eE+-]+;", "endTimeValue        0;",
                           backup, count=1, flags=re.M)
            ctrl.write_text(probe)
            cmd = (f"mpirun --oversubscribe -np {self.n_cores} pimpleFoam -parallel")
            bashrc = self.find_openfoam_bashrc()
            full = f"bash -c 'source {bashrc} && cd {self.case_dir} && {cmd}'" if bashrc \
                else f"bash -c 'cd {self.case_dir} && {cmd}'"
            res = subprocess.run(full, shell=True, capture_output=True,
                                 text=True, timeout=300)
            txt = (res.stdout or "") + (res.stderr or "")
            m = None
            for m2 in re.finditer(r"Courant Number mean:\s*([0-9.eE+-]+)\s+max:\s*([0-9.eE+-]+)", txt):
                m = m2
                break     # 첫 번째(= 설정 deltaT 기준) 값만 쓴다
            if not m:
                out["reason"] = "Courant 수를 읽지 못했습니다"
                return out
            co_max = float(m.group(2))
            out["co_max"] = co_max
            if co_max <= 0:
                out["reason"] = "Courant 수가 0"
                return out
            dt = float(delta_t) * float(max_co) / co_max
            dt = min(dt, float(delta_t))      # 설정값보다 커지지는 않게
            out["delta_t"] = dt
            out["steps"] = duration / dt if dt > 0 else float("inf")
            out["ok"] = True

            # ── 스텝당 실측 소요시간 ──────────────────────────────────────
            # 종전에는 호출부가 0.4초/스텝 을 가정해 소요시간을 알렸는데, 실제와
            # 최대 8배까지 어긋났다(196만 셀 실측 3.5초/스텝 → '2.2시간' 안내가
            # 실제로는 17시간). 몇 스텝만 실제로 돌려 ExecutionTime 증가분의
            # 중앙값으로 잰다(시작 오버헤드는 증가분을 쓰므로 제외된다).
            # 실패해도 진단 자체는 유효하므로 조용히 넘어간다(호출부가 fallback).
            try:
                probe2 = re.sub(r"^endTimeValue\s+[0-9.eE+-]+;",
                                f"endTimeValue        {dt * 4:.12g};",
                                backup, count=1, flags=re.M)
                ctrl.write_text(probe2)
                res2 = subprocess.run(full, shell=True, capture_output=True,
                                      text=True, timeout=600)
                txt2 = (res2.stdout or "") + (res2.stderr or "")
                ex = [float(x) for x in
                      re.findall(r"^ExecutionTime = ([0-9.eE+-]+) s", txt2, re.M)]
                if len(ex) >= 3:
                    dd = sorted(ex[i] - ex[i - 1] for i in range(1, len(ex)))
                    out["sec_per_step"] = dd[len(dd) // 2]
            except Exception:
                pass
            return out
        except subprocess.TimeoutExpired:
            out["reason"] = "프로브 실행 시간 초과"
            return out
        except Exception as _e:
            out["reason"] = f"프로브 실패: {_e}"
            return out
        finally:
            ctrl.write_text(backup)

    def _reset_processor_time_to_zero(self, latest: float) -> None:
        """각 processorN/ 의 최신 시간 디렉토리를 0 으로 옮긴다.

        simpleFoam 수렴장을 pimpleFoam 의 초기조건(t=0)으로 삼기 위한 것.
        기존 0/ 은 초기 균일장이라 버려도 된다(수렴장이 더 좋은 초기조건).
        forceCoeffs/forces 이력도 반복 기반이라 함께 지워 통계 오염을 막는다.
        """
        moved = 0
        for pdir in sorted(self.case_dir.glob("processor*")):
            src = None
            for d in pdir.iterdir():
                if not d.is_dir():
                    continue
                try:
                    if abs(float(d.name) - latest) < 1e-9:
                        src = d
                        break
                except ValueError:
                    continue
            if src is None:
                continue
            dst = pdir / "0"
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                src.rename(dst)
                # [중요] <time>/uniform/time 에는 이전 시간값(예 300)과 simpleFoam
                # 의 deltaT(=1초)가 들어 있고, 이것이 controlDict 의 startTime·
                # deltaT 를 덮어쓴다. 그대로 두면 Courant 수가 1e4 대로 치솟고
                # 'endTime < 현재시간' 이 되어 시간 루프가 즉시 종료된다.
                # 지우면 OpenFOAM 이 controlDict 값을 그대로 쓴다.
                shutil.rmtree(dst / "uniform", ignore_errors=True)
                moved += 1
            except Exception as _e:
                self._emit_log(f"⚠️ {pdir.name} 시간 재설정 실패: {_e}")
        # 반복 기반 force 이력 제거(물리시간 이력과 섞이면 통계가 오염된다)
        pp = self.case_dir / "postProcessing"
        if pp.exists():
            for sub in list(pp.glob("force*")):
                shutil.rmtree(sub, ignore_errors=True)
        self._emit_log(f"수렴장을 t=0 으로 이동 ({moved}개 processor) · "
                       f"반복 기반 force 이력 제거")

    def _latest_processor_time(self) -> float:
        """processor0/ 의 최신 시간 디렉토리 값(없으면 0)."""
        p0 = self.case_dir / "processor0"
        if not p0.exists():
            return 0.0
        times = []
        for d in p0.iterdir():
            if d.is_dir():
                try:
                    times.append(float(d.name))
                except ValueError:
                    pass
        return max(times) if times else 0.0

    # ─── 전체 워크플로우 실행 ─────────────────────────────────────────────

    def run_full_workflow(self, end_time: int = 2000,
                          transient: Optional[Dict[str, Any]] = None) -> bool:
        """전체 파이프라인. transient=None 이면 종전과 100% 동일한 steady 경로.

        transient(dict) 가 주어지면 pimpleFoam 경로로 분기한다:
          init_from_steady=True  → simpleFoam 선행 수렴 후 그 장을 초기조건으로 사용
          perturb=True           → 대칭 형상의 와류 유발을 위한 미소 비대칭 교란(보완③)
        """
        # 필수 단계(여기 실패하면 해석 자체 실패)
        essential = [
            self.run_blockMesh,
            self.run_surfaceFeatureExtract,
            self.run_snappyHexMesh,
        ]
        for step_fn in essential:
            if self._stop_flag.is_set():
                return False
            if not step_fn():
                return False
        # 솔버 (Cd/Cl·필드 생성)
        if self._stop_flag.is_set():
            return False
        if transient:
            if not self._run_transient(end_time, transient):
                return False
        elif not self.run_solver(end_time):
            return False
        # reconstructPar는 '시각화용 편의' 단계 — 실패해도 forceCoeffs(Cd/Cl)는
        # postProcessing에 이미 있으므로 결과 추출에는 지장 없음. best-effort 처리.
        try:
            if not self.run_reconstructPar():
                self._emit_log("⚠️ reconstructPar 실패(시각화용) — Cd/Cl 추출은 계속 진행")
        except Exception as _e:
            self._emit_log(f"⚠️ reconstructPar 예외(무시): {_e}")
        return True

    def stop(self):
        """해석 중지"""
        self._stop_flag.set()
        if self._proc:
            self._proc.terminate()

    # ─── 내부 실행 헬퍼 ───────────────────────────────────────────────────

    def _emit_step(self, module: str, status: str, pct: float, detail: str = ""):
        """step_cb 호출 헬퍼"""
        if self.step_cb:
            try:
                self.step_cb(module, status, pct, detail)
            except Exception:
                pass

    def _run_step(self, cmd: str, label: str,
                  parallel: bool = False,
                  pre_cmd: Optional[str] = None,
                  monitor_residuals: bool = False,
                  end_time: int = 2000,
                  start_time: float = 0.0) -> bool:
        if self._stop_flag.is_set():
            return False

        if pre_cmd:
            self._run_step(pre_cmd, f"사전 단계: {pre_cmd}")

        full_cmd = self._of_cmd(cmd)
        self._emit_log(f"\n{'='*50}\n▶ {label} 시작\n{'='*50}")
        self._emit_step(label, "running", 0, "시작")

        # snappyHexMesh 단계 추적용 (단계별 반복 횟수로 진행률을 동적으로 갱신)
        _snappy_pct  = 0.0
        _cast_iter   = 0   # castellation refinement 반복
        _morph_iter  = 0   # snapping(morph) 반복
        _layer_iter  = 0   # layer addition 반복
        import time as _time
        _snap_t0     = _time.time()

        try:
            self._proc = subprocess.Popen(
                full_cmd, shell=True, executable="/bin/bash",
                cwd=str(self.case_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "MPI_NUM_PROCS": str(self.n_cores)}
            )

            step = 0
            with open(self.log_file, "a") as lf:
                for line in self._proc.stdout:
                    lf.write(line)
                    self._emit_log(line.rstrip())

                    if monitor_residuals:
                        self._parse_residuals(line, step, end_time)
                        # 보완①: 종전 패턴은 r"^Time = (\d+)" 로 '정수'만 매치했다.
                        # simpleFoam 은 반복 횟수(정수)를 찍지만 pimpleFoam 은
                        # 'Time = 0.0025' 처럼 부동소수 물리시간을 찍으므로 매치에
                        # 실패해 진행률이 0% 에 고착되고 ETA 가 무력화됐다.
                        # 부동소수·지수표기를 모두 받도록 확장한다. simpleFoam 은
                        # 정수가 그대로 float 로 파싱되어 동작이 종전과 동일하다.
                        m = re.match(r"^Time = ([0-9.eE+-]+)", line)
                        if m:
                            try:
                                step = float(m.group(1))
                            except ValueError:
                                step = 0.0
                            # 이어받기(simpleFoam 수렴해에서 pimpleFoam 계속)면
                            # 시작 시각이 0 이 아니다. start_time 을 빼지 않으면
                            # 시작하자마자 진행률이 97% 로 보인다.
                            _span = max(float(end_time) - float(start_time), 1e-12)
                            pct = round(min(max((step - float(start_time)) / _span, 0.0)
                                            * 100, 99.9), 1)
                            residual_info = ""
                            if self.last_residuals:
                                max_r = max(self.last_residuals.values())
                                residual_info = f"잔차 {max_r:.2e}"
                            # 정수면 종전 표기(반복 횟수), 소수면 물리시간 표기.
                            _is_int = (float(step).is_integer()
                                       and float(end_time).is_integer()
                                       and float(start_time) == 0.0)
                            _cur = f"{int(step)}" if _is_int else f"{step:g}s"
                            _tot = f"{int(end_time)}" if _is_int else f"{float(end_time):g}s"
                            self._emit_step(label, "running", pct,
                                            f"Time={_cur}/{_tot}  {residual_info}")
                            if self.progress_cb:
                                self.progress_cb(pct, step, end_time)

                    # snappyHexMesh 단계 파싱 — 반복 횟수로 진행률을 동적으로 움직인다
                    elif "snappyHexMesh" in cmd or "snappyHexMesh" in label:
                        _line = line.strip().lower()
                        _el   = int(_time.time() - _snap_t0)           # 경과(초)
                        _elap = f"{_el//60}분 {_el%60:02d}초"
                        if "writing mesh" in _line or "written mesh" in _line:
                            # 실제 최종 격자 저장
                            _snappy_pct = max(_snappy_pct, 95.0)
                            self._emit_step(label, "running", _snappy_pct,
                                            f"격자 저장 중... ({_elap})")
                        elif "morph iteration" in _line or "moving mesh" in _line:
                            # 스내핑(표면 적합) 단계 — Morph iteration 반복
                            if "morph iteration" in _line:
                                _morph_iter += 1
                            _snappy_pct = min(70.0, max(_snappy_pct, 40.0 + _morph_iter * 2.0))
                            self._emit_step(label, "running", _snappy_pct,
                                            f"표면 적합(스내핑) {_morph_iter}회 반복 중 · {_elap}")
                        elif "add layer" in _line or "layer addition" in _line \
                                or ("layer" in _line and "iteration" in _line):
                            _layer_iter += 1
                            _snappy_pct = min(92.0, max(_snappy_pct, 72.0 + _layer_iter * 2.0))
                            self._emit_step(label, "running", _snappy_pct,
                                            f"경계층 추가 {_layer_iter}회 · {_elap}")
                        elif "refinement iteration" in _line or "castellat" in _line:
                            if "refinement iteration" in _line:
                                _cast_iter += 1
                            _snappy_pct = min(38.0, max(_snappy_pct, 8.0 + _cast_iter * 6.0))
                            self._emit_step(label, "running", _snappy_pct,
                                            f"격자 세분화 {_cast_iter}회 · {_elap}")

            self._proc.wait()
            success = (self._proc.returncode == 0)
            status = "✅ 완료" if success else "❌ 실패"
            self._emit_log(f"\n{status}: {label}")
            self._emit_step(label, "done" if success else "error", 100 if success else 0, "")
            return success

        except Exception as e:
            self._emit_log(f"❌ 오류 발생: {e}")
            self._emit_step(label, "error", 0, str(e))
            return False

    def _parse_residuals(self, line: str, step: int, end_time: int):
        """잔차 파싱 및 수렴 판정"""
        # 예: "smoothSolver:  Solving for Ux, Initial residual = 1.23e-05, ..."
        m = re.search(r"Solving for (\w+),\s+Initial residual = ([\d.e+-]+)", line)
        if m:
            field = m.group(1)
            resid = float(m.group(2))
            self.last_residuals[field] = resid

        # 수렴 판정
        if len(self.last_residuals) >= 3:
            max_r = max(self.last_residuals.values())
            if max_r < 1e-4:
                self.converged = True

    def _emit_log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)

    def _write_surfaceFeatureExtractDict(self):
        """surfaceFeatureExtractDict 자동 생성"""
        sfe_dir = self.case_dir / "system"

        # STL 파일 목록 수집
        trisurf = self.case_dir / "constant" / "triSurface"
        stl_files = list(trisurf.glob("*.stl")) if trisurf.exists() else []

        entries = ""
        for stl in stl_files:
            entries += f"""
    {stl.name}
    {{
        extractionMethod    extractFromSurface;
        extractFromSurfaceCoeffs
        {{
            includedAngle   150;
        }}
        writeObj    yes;
    }}
"""
        content = f"""FoamFile
{{
    version 2.0; format ascii;
    class dictionary; object surfaceFeatureExtractDict;
}}
{entries}
"""
        (sfe_dir / "surfaceFeatureExtractDict").write_text(content)


# ═══════════════════════════════════════════════════════════════════════════
# 결과 추출 및 CSV 저장
# ═══════════════════════════════════════════════════════════════════════════

class ResultExtractor:
    """
    해석 결과에서 Cd, Cl 추출 → CSV 저장
    질량-스프링 모델과 호환되는 표준 포맷
    """

    HEADER = [
        "speed_m_s", "angle_deg",
        "Cd", "Cl", "Cm",
        "Fx_N", "Fy_N", "Fz_N",
        "rho_kg_m3", "case_name", "timestamp",
        # 요구서 §10·§21: CD/CL 계산에 '실제로 쓰인' 기준값을 결과에 남긴다.
        # 하위 C++ 모델이 열 이름으로 파싱하므로 기존 열 순서는 건드리지 않고
        # 뒤에 덧붙인다.
        "Aref_m2", "lRef_m", "nu_m2_s",
    ]

    def __init__(self, case_dir: Path, speed: float, angle_deg: float,
                 rho: float = 1025.0):
        self.case_dir  = case_dir
        self.speed     = speed
        self.angle_deg = angle_deg
        self.rho       = rho

    def extract_force_coeffs(self) -> Optional[Dict]:
        """postProcessing/forceCoeffs 디렉토리에서 최종 값 추출"""
        pp_dir = self.case_dir / "postProcessing"
        force_dirs = list(pp_dir.glob("forceCoeffs*"))
        if not force_dirs:
            # 단위 셀 모드
            force_dirs = list(pp_dir.glob("forceCoeffs"))

        results = {}
        for fd in force_dirs:
            time_dirs = sorted(
                [d for d in fd.iterdir() if d.is_dir()],
                key=lambda x: float(x.name) if x.name.replace('.','').isdigit() else 0
            )
            if not time_dirs:
                continue

            # 최신 결과 파일
            coeff_file = time_dirs[-1] / "forceCoeffs.dat"
            if not coeff_file.exists():
                coeff_file = time_dirs[-1] / "coefficient.dat"
            if not coeff_file.exists():
                continue

            # 마지막 10줄 평균 (수렴값 추출)
            lines = [l for l in coeff_file.read_text().splitlines()
                     if not l.startswith("#") and l.strip()]
            if not lines:
                continue

            last_lines = lines[-min(10, len(lines)):]
            cd_vals, cl_vals, cm_vals = [], [], []
            for line in last_lines:
                parts = line.split()
                # coefficient.dat 컬럼: Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch ...
                if len(parts) >= 8:
                    try:
                        cd_vals.append(float(parts[1]))  # Cd (total)
                        cl_vals.append(float(parts[4]))  # Cl (total)
                        cm_vals.append(float(parts[7]))  # CmPitch
                    except (ValueError, IndexError):
                        pass

            if cd_vals:
                results = {
                    "Cd": sum(cd_vals) / len(cd_vals),
                    "Cl": sum(cl_vals) / len(cl_vals),
                    "Cm": sum(cm_vals) / len(cm_vals),
                }
        return results if results else None

    def extract_forces(self) -> Optional[Dict]:
        """forces 함수 객체의 출력에서 합력(N)을 추출한다.

        ESI v2312는 `postProcessing/forces/<time>/force.dat` 에 합력을 쓴다.
        컬럼은 `Time  total_x total_y total_z  pressure_xyz  viscous_xyz`
        형식이며, 버전에 따라 벡터가 괄호 `(...)` 로 묶여 나올 수도 있어
        괄호를 제거한 뒤 파싱한다(legacy `forces.dat`도 같은 방식으로 처리).
        마지막 10스텝 평균을 합력으로 본다(계수 추출과 동일 정책).
        """
        pp_dir = self.case_dir / "postProcessing"
        force_dirs = list(pp_dir.glob("forces*"))
        for fd in force_dirs:
            time_dirs = sorted(
                [d for d in fd.iterdir() if d.is_dir()],
                key=lambda x: float(x.name) if x.name.replace('.', '').isdigit() else 0
            )
            if not time_dirs:
                continue
            # ESI v2312: force.dat / legacy: forces.dat
            f_file = time_dirs[-1] / "force.dat"
            if not f_file.exists():
                f_file = time_dirs[-1] / "forces.dat"
            if not f_file.exists():
                continue

            lines = [l for l in f_file.read_text().splitlines()
                     if not l.startswith("#") and l.strip()]
            if not lines:
                continue

            last_lines = lines[-min(10, len(lines)):]
            fx_vals, fy_vals, fz_vals = [], [], []
            for line in last_lines:
                # 괄호로 묶인 벡터 표기 `( ... )` 를 제거해 평탄한 컬럼으로 만든다.
                parts = line.replace("(", " ").replace(")", " ").split()
                # parts[0]=Time, parts[1:4]=총합력(total) x/y/z
                if len(parts) >= 4:
                    try:
                        fx_vals.append(float(parts[1]))
                        fy_vals.append(float(parts[2]))
                        fz_vals.append(float(parts[3]))
                    except (ValueError, IndexError):
                        pass

            if fx_vals:
                n = len(fx_vals)
                return {
                    "Fx_N": sum(fx_vals) / n,
                    "Fy_N": sum(fy_vals) / n,
                    "Fz_N": sum(fz_vals) / n,
                }
        return None

    def is_transient_case(self) -> bool:
        """이 케이스가 pimpleFoam 으로 돌았는지(controlDict 의 application 기준)."""
        try:
            txt = (self.case_dir / "system" / "controlDict").read_text()
            m = re.search(r"application\s+(\w+);", txt)
            return bool(m and m.group(1) == "pimpleFoam")
        except Exception:
            return False

    def save_csv(self, output_path: Path,
                 t_avg_start: Optional[float] = None) -> Path:
        """결과를 CSV 파일로 저장 (질량-스프링 모델 호환 포맷).

        지시서 §22 방침: CSV 의 열 구성·열 이름은 절대 바꾸지 않는다(하위 C++ 모델
        호환). 다만 transient 케이스는 Cd/Cl/Fx/Fy/Fz 에 '마지막 값'이 아니라
        '평균구간 시간평균값'을 넣어, steady 결과와 같은 방식으로 소비되게 한다.
        시간이력·RMS·평균구간 등 부가 정보는 transient_stats.json 에 따로 저장한다.
        """
        coeffs = self.extract_force_coeffs() or {}
        forces = self.extract_forces() or {}

        if self.is_transient_case():
            stats = compute_transient_stats(self.case_dir, t_avg_start=t_avg_start)
            if stats.get("n_samples"):
                # 시간평균값으로 치환(없는 항목은 기존 마지막 값 유지)
                for _k, _sk in (("Cd", "mean_Cd"), ("Cl", "mean_Cl")):
                    if _sk in stats:
                        coeffs[_k] = stats[_sk]
                for _k, _sk in (("Fx_N", "mean_Fx"), ("Fy_N", "mean_Fy"),
                                ("Fz_N", "mean_Fz")):
                    if _sk in stats:
                        forces[_k] = stats[_sk]
                stats["solver"] = "pimpleFoam"
                stats["speed_m_s"] = self.speed
                stats["angle_deg"] = self.angle_deg
                try:
                    (self.case_dir / "transient_stats.json").write_text(
                        json.dumps(stats, ensure_ascii=False, indent=2, default=str))
                except Exception as _e:
                    logger.warning(f"transient_stats.json 저장 실패: {_e}")
                logger.info(
                    f"[Transient] CSV 에 시간평균 기록: Cd={coeffs.get('Cd')} "
                    f"(평균구간 {stats.get('t_avg_start')}~{stats.get('t_end')}s, "
                    f"샘플 {stats.get('n_samples')}개, "
                    f"변동/평균 {stats.get('unsteadiness_Cd')})")
            else:
                logger.warning("[Transient] 시간이력이 없어 마지막 값으로 기록")

        row = {
            "speed_m_s":  self.speed,
            "angle_deg":  self.angle_deg,
            "Cd":         coeffs.get("Cd", float("nan")),
            "Cl":         coeffs.get("Cl", float("nan")),
            "Cm":         coeffs.get("Cm", float("nan")),
            "Fx_N":       forces.get("Fx_N", float("nan")),
            "Fy_N":       forces.get("Fy_N", float("nan")),
            "Fz_N":       forces.get("Fz_N", float("nan")),
            "rho_kg_m3":  self.rho,
            "case_name":  self.case_dir.name,
            "timestamp":  datetime.now().isoformat(),
        }

        # 케이스가 실제로 사용한 기준값(자동 계산·직접 입력·재실행 무관하게 일치)
        _ref = read_case_reference(self.case_dir)
        row["Aref_m2"] = _ref.get("Aref_m2")
        row["lRef_m"]  = _ref.get("lRef_m")
        try:
            _tp = (self.case_dir / "constant" / "transportProperties").read_text()
            _m = re.search(r"^\s*nu\s+([0-9.eE+-]+)\s*;", _tp, re.M)
            row["nu_m2_s"] = float(_m.group(1)) if _m else None
        except Exception:
            row["nu_m2_s"] = None
        # rhoInf 는 케이스 값이 우선(요구서 §21 — 계산에 쓰인 값을 남긴다)
        if _ref.get("rhoInf"):
            row["rho_kg_m3"] = _ref["rhoInf"]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 항목9(덮어쓰기): 같은 (유속, 영각) 조건의 기존 행은 제거하고 새 행으로 교체한다.
        # 단순 append 는 재해석 시 동일 조건 중복 행을 남겨 표시 유속·Cd/Cl 요약을
        # 오염시키므로, 프로젝트 CSV는 조건당 1행만 유지(최신 결과로 덮어씀).
        _existing = []
        if output_path.exists():
            try:
                with open(output_path, newline="") as f:
                    for r in csv.DictReader(f):
                        try:
                            _same = (abs(float(r.get("speed_m_s", "nan")) - float(self.speed)) < 1e-6
                                     and abs(float(r.get("angle_deg", "nan")) - float(self.angle_deg)) < 1e-6)
                        except (TypeError, ValueError):
                            _same = False
                        if not _same:
                            _existing.append(r)
            except Exception:
                _existing = []

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            for r in _existing:
                writer.writerow({k: r.get(k, "") for k in self.HEADER})
            writer.writerow(row)

        logger.info(f"결과 저장(조건당 1행 유지): {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════════════
# 배치 해석 관리자 (영각/유속 자동 순환)
# ═══════════════════════════════════════════════════════════════════════════

def count_mesh_cells(case_dir: Path) -> Optional[int]:
    """생성된 격자의 셀 수를 읽는다(polyMesh/owner 헤더의 nCells).

    병렬 실행에서는 실제 계산에 쓰인 격자가 processor*/ 에 있다. 케이스 루트의
    constant/polyMesh 는 재구성 시점·경로에 따라 실제 격자와 다를 수 있으므로
    processor 격자가 있으면 그 합을 우선한다.
    """
    procs = sorted(case_dir.glob("processor*/constant/polyMesh/owner"))
    if procs:
        total = 0
        for p in procs:
            try:
                m = re.search(r"nCells:\s*(\d+)", p.read_text(errors="ignore")[:4000])
                if m:
                    total += int(m.group(1))
            except Exception:
                continue
        if total > 0:
            return total
    try:
        head = (case_dir / "constant" / "polyMesh" / "owner").read_text(errors="ignore")[:4000]
        m = re.search(r"nCells:\s*(\d+)", head)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def mesh_independence_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """격자 독립성 결과에 직전 격자 대비 변화율을 붙인다(요구서 §19)."""
    out: List[Dict[str, Any]] = []
    prev_cd = prev_cl = None
    for r in sorted(rows, key=lambda x: (x.get("cells") or 0)):
        cd, cl = r.get("Cd"), r.get("Cl")
        d_cd = (None if (prev_cd in (None, 0) or cd is None)
                else (cd - prev_cd) / abs(prev_cd) * 100.0)
        d_cl = (None if (prev_cl in (None, 0) or cl is None)
                else (cl - prev_cl) / abs(prev_cl) * 100.0)
        out.append({**r, "dCd_pct": d_cd, "dCl_pct": d_cl})
        if cd is not None:
            prev_cd = cd
        if cl is not None:
            prev_cl = cl
    return out


def run_mesh_independence(mode: str, stl_paths: Dict[str, Path],
                          speed: float, angle: float, levels: List[int],
                          common_params: Dict[str, Any], results_root: Path,
                          log_cb: Optional[Callable] = None,
                          progress_cb: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """같은 물리조건에서 정밀화 레벨만 바꿔 연속 실행한다(요구서 §19).

    기존 BatchAnalysisManager 를 레벨마다 한 번씩 쓰는 방식이라 실행 경로가
    통상 해석과 완전히 같다. 결과에 셀 수·Cd·Cl 과 직전 대비 변화율을 낸다.
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for i, lv in enumerate(levels):
        if log_cb:
            log_cb(f"▶ 격자 독립성 {i+1}/{len(levels)} — 정밀화 레벨 {lv}")
        sub = results_root / f"lv{lv}"
        sub.mkdir(parents=True, exist_ok=True)
        params = dict(common_params)
        params["refine_level"] = int(lv)
        params["auto_refine"] = False      # 레벨을 고정해야 비교가 성립한다
        mgr = BatchAnalysisManager(
            mode=mode, stl_paths=stl_paths, speeds=[speed], angles=[angle],
            output_csv=sub / "force_coeffs.csv", common_params=params,
            progress_cb=(lambda p, s, e, label="", _i=i:
                         progress_cb((_i + p / 100.0) / len(levels) * 100.0,
                                     _i + 1, len(levels), label=f"레벨 {lv} — {label}")
                         if progress_cb else None),
            log_cb=log_cb, results_root=sub)
        try:
            mgr.run_batch()
        except Exception as e:
            if log_cb:
                log_cb(f"❌ 레벨 {lv} 실패: {e}")
            rows.append({"level": lv, "cells": None, "Cd": None, "Cl": None,
                         "error": str(e)})
            continue
        # run_batch() 의 반환값은 비어 있다(결과의 정본은 CSV 다). CSV 를 읽는다.
        cd = cl = None
        _csv = sub / "force_coeffs.csv"
        if _csv.exists():
            try:
                _rows = list(csv.DictReader(open(_csv)))
                if _rows:
                    cd = float(_rows[-1]["Cd"])
                    cl = float(_rows[-1]["Cl"])
            except Exception:
                pass
        cells = None
        for c in sorted(sub.glob("해석완료_*")):
            cells = count_mesh_cells(c) or cells
        rows.append({"level": lv, "cells": cells, "Cd": cd, "Cl": cl})
        if log_cb:
            log_cb(f"  레벨 {lv}: 셀 {cells if cells else '?'} · "
                   f"Cd={cd if cd is None else round(cd, 5)}")
    return mesh_independence_table(rows)


# ── 검증용 Reference case (요구서 §20) ────────────────────────────────────
# 문헌값을 프로그램에 담을 때는 출처와 조건을 반드시 함께 남긴다(요구서 §20).
# 아래 값은 교과서·고전 실험의 표준값이며, 본 프로그램이 재현을 보증하는 값이
# 아니다. 실제로 구는 정상 RANS 로 재현되지 않는다는 것을 실측으로 확인했다.
REFERENCE_CASES: Dict[str, Dict[str, Any]] = {
    "sphere": {
        "label": "구 (Sphere)",
        "stl": "Sphere.stl",
        "length_m": 0.300,
        "Cd_ref": 0.50,
        "Re_range": (1.0e4, 3.0e5),   # 항력위기(Re≈3e5) 직전까지 아임계
        "source": "Achenbach, E. (1972) J. Fluid Mech. 54:565 / "
                  "Schlichting, Boundary-Layer Theory (8th ed.) — 매끈한 구, "
                  "아임계 영역 Cd ≈ 0.5 (항력위기 Re≈3e5 이전)",
        "note": "본 프로그램 실측: simpleFoam Cd=0.228, DDES Cd=0.218 로 문헌값의 "
                "절반 수준이다. 구는 박리점이 표면 위를 움직여 정상 RANS 로 재현이 "
                "어려운 형상이며, 이 케이스는 '재현되지 않는다는 사실'을 확인하는 "
                "용도로 쓴다.",
    },
    "cylinder": {
        "label": "원기둥 (Cylinder, L/D=4)",
        "stl": None,          # 없으면 생성한다
        "length_m": 0.050,
        "Cd_ref": 0.80,
        "Re_range": (1.0e4, 2.0e5),
        "source": "Hoerner, S.F. (1965) Fluid-Dynamic Drag, Ch.3 — 유한 길이 "
                  "원기둥(L/D≈4)의 Cd ≈ 0.8. 무한 길이(2D) 기준값은 "
                  "Wieselsberger (1921) 의 Cd ≈ 1.2 이며 서로 다른 값이다.",
        "note": "끝면 효과 때문에 유한 원기둥은 2D 값보다 낮다. 비교할 때 어느 "
                "기준값을 쓰는지 반드시 맞춰야 한다.",
    },
}


def write_cylinder_stl(path: Path, diameter_mm: float = 50.0,
                       length_mm: float = 200.0, n_seg: int = 96) -> Path:
    """검증용 원기둥 STL 을 생성한다(축 = x). 닫힌 매니폴드."""
    R, L = diameter_mm / 2.0, length_mm
    tris = []
    for i in range(n_seg):
        a0 = 2 * math.pi * i / n_seg
        a1 = 2 * math.pi * (i + 1) / n_seg
        p0 = (0.0, R*math.cos(a0), R*math.sin(a0))
        p1 = (0.0, R*math.cos(a1), R*math.sin(a1))
        q0 = (L, R*math.cos(a0), R*math.sin(a0))
        q1 = (L, R*math.cos(a1), R*math.sin(a1))
        tris += [(p0, q0, q1), (p0, q1, p1),
                 ((0.0, 0.0, 0.0), p1, p0), ((L, 0.0, 0.0), q0, q1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("solid cylinder\n")
        for t in tris:
            f.write(" facet normal 0 0 0\n  outer loop\n")
            for v in t:
                f.write(f"   vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("  endloop\n endfacet\n")
        f.write("endsolid cylinder\n")
    return path


def reference_comparison(name: str, cd_cfd: float, speed: float,
                         nu: float) -> Dict[str, Any]:
    """검증 케이스 결과를 문헌값과 비교한다. 출처·조건을 함께 반환한다."""
    ref = REFERENCE_CASES.get(name)
    if not ref:
        return {"ok": False}
    L = ref["length_m"]
    re_num = speed * L / nu if nu > 0 else 0.0
    lo, hi = ref["Re_range"]
    return {
        "ok": True, "label": ref["label"], "Cd_ref": ref["Cd_ref"],
        "Cd_cfd": cd_cfd, "Re": re_num, "Re_in_range": lo <= re_num <= hi,
        "error_pct": ((cd_cfd - ref["Cd_ref"]) / ref["Cd_ref"] * 100.0
                      if ref["Cd_ref"] else None),
        "source": ref["source"], "note": ref["note"],
        "length_m": L,
    }


class BatchAnalysisManager:
    """
    영각 × 유속 조합 배치 해석 자동 실행
    - 파라미터 매트릭스 자동 생성
    - 순차 실행 및 결과 집계
    - 진행률 스트리밍
    """

    def __init__(self, mode: str,
                 stl_paths: Dict[str, Path],
                 speeds: List[float],
                 angles: List[float],
                 output_csv: Path,
                 common_params: Optional[Dict] = None,
                 progress_cb: Optional[Callable] = None,
                 log_cb: Optional[Callable] = None,
                 results_root: Optional[Path] = None):
        self.mode        = mode          # "unit_cell" or "full_structure"
        self.stl_paths   = stl_paths
        self.speeds      = speeds
        self.angles      = angles
        self.output_csv  = output_csv
        self.params      = common_params or {}
        self.progress_cb = progress_cb
        self.log_cb      = log_cb
        # 케이스 디렉토리를 둘 루트. 프로젝트 폴더(있으면)로 라우팅, 없으면 모드 루트.
        self.results_root = Path(results_root) if results_root else (RESULTS_DIR / mode)
        self._stop_flag  = threading.Event()
        self.results: List[Dict] = []
        # 실제 성공/실패 케이스 수(정직한 완료 보고용 — '완료'인데 결과 0 방지)
        self.n_success = 0
        self.n_failed = 0

    def run_batch(self) -> List[Dict]:
        """배치 해석 실행"""
        combos = [(s, a) for s in self.speeds for a in self.angles]
        total  = len(combos)

        for i, (speed, angle) in enumerate(combos):
            if self._stop_flag.is_set():
                break

            self._log(f"\n{'='*60}")
            self._log(f"배치 {i+1}/{total}: 유속={speed:.2f}m/s, 영각={angle:.1f}°")
            self._log(f"{'='*60}")

            # 케이스 디렉토리 명명: 실행 중에는 '해석중_', 완료 후 '해석완료_'로
            # rename 한다(항목 12). 결과분석 매트릭스가 이 접두어로 상태를 구분한다.
            base_name    = f"{self.mode}_U{speed:.2f}_A{angle:.1f}"
            running_name = f"해석중_{base_name}"
            done_name    = f"해석완료_{base_name}"
            case_dir  = self.results_root / running_name
            done_dir  = self.results_root / done_name
            # 같은 조합의 이전 '해석중_' 잔여물 정리(완료본은 성공 시점에만 교체)
            if case_dir.exists():
                shutil.rmtree(case_dir, ignore_errors=True)

            # 케이스 시작 직전 — 초기화/격자생성 구간에도 진행률 표시
            self._progress(round(i / total * 100, 1), i + 1, total)

            try:
                if self.mode == "unit_cell":
                    builder = UnitCellCaseBuilder(
                        case_dir=case_dir,
                        stl_path=self.stl_paths.get("net"),
                        speed=speed, angle_deg=angle,
                        # 계산량 옵션(end_time·refine_level 등)도 전달해야 배치가
                        # 프리셋(최소/보통/정밀)을 반영한다. (이전엔 무시되어 항상 기본값)
                        **{k: v for k, v in self.params.items()
                           if k in ["cell_size", "n_cores", "nx", "ny",
                                    "residual_control", "end_time",
                                    "write_interval", "refine_level", "solidity",
                                    "aref_override", "rho", "nu"]}
                    )
                else:
                    builder = FullStructureCaseBuilder(
                        case_dir=case_dir,
                        cage_stl=self.stl_paths.get("cage"),
                        net_stl=self.stl_paths.get("net"),
                        speed=speed, angle_deg=angle,
                        # end_time/residual/write_interval 도 전달해야 한다. 누락 시
                        # 빌더 기본값(end_time=3000)으로 케이스가 생성돼, 진행바가
                        # 쓰는 프리셋 end_time(예: 500)과 어긋나 솔버가 3000회까지 도는
                        # 동안 진행바가 99.9%에 고착된다(케이스가 멈춘 것처럼 보임).
                        **{k: v for k, v in self.params.items()
                           if k in ["cage_diameter", "cage_depth", "n_cores",
                                    "end_time", "residual_control", "write_interval",
                                    "aref_override", "refine_level", "n_layers",
                                    "auto_refine", "net_grid_redesign",
                                    "net_grid_target_cells", "wake_box_level",
                                    "rho", "nu"]}
                    )

                builder.build()

                # OpenFOAM 단계별 진행을 배치 UI에도 전달하는 step_cb
                _i = i
                def _batch_step_cb(module, status, pct, detail, _ii=_i):
                    label_map = {
                        "blockMesh":        "격자 생성",
                        "snappyHexMesh":    "격자 스냅",
                        "simpleFoam":       "CFD 해석",
                        "reconstructPar":   "결과 재조합",
                    }
                    short = next(
                        (v for k, v in label_map.items() if k in module), module
                    )
                    step_desc = f"{detail}" if detail else status
                    self._progress(
                        round((_ii / total) * 100 + pct / total, 1),
                        _ii + 1, total,
                        label=f"케이스 {_ii+1}/{total} — {short} {step_desc}"
                    )

                n_cores = self.params.get("n_cores", get_cpu_count())
                runner = OpenFOAMRunner(
                    case_dir=case_dir,
                    n_cores=n_cores,
                    progress_cb=lambda p, s, e, _ii=_i: self._progress(
                        round((_ii + p / 100) / total * 100, 1), _ii + 1, total
                    ),
                    log_cb=self.log_cb,
                    step_cb=_batch_step_cb,
                )
                # Solver 분기: transient 가 없으면 종전과 동일한 steady 경로.
                # transient 면 end_time 은 '물리시간[s]' 이라 int 로 깎으면 안 된다.
                _tr = self.params.get("transient")
                if _tr:
                    _et = float(_tr.get("end_time", 30.0))
                    ok = runner.run_full_workflow(end_time=_et, transient=_tr)
                else:
                    _et = int(self.params.get("end_time", 2000))
                    ok = runner.run_full_workflow(end_time=_et)

                # 워크플로우 반환값과 무관하게 forceCoeffs(Cd/Cl)가 있으면 추출한다.
                # (reconstructPar나 솔버의 비치명적 비정상 종료로 ok=False여도 결과가
                #  생성됐으면 살린다 → 결과 유실 방지)
                _cf  = ResultExtractor(case_dir, speed, angle).extract_force_coeffs() or {}
                _cdv = _cf.get("Cd")
                if isinstance(_cdv, (int, float)):
                    # 성공 → '해석완료_'로 rename 후 그 경로로 CSV 저장(case_name 반영)
                    final_dir = case_dir
                    try:
                        if done_dir.exists():
                            shutil.rmtree(done_dir, ignore_errors=True)
                        case_dir.rename(done_dir)
                        final_dir = done_dir
                    except Exception as _re:
                        self._log(f"⚠️ 완료 rename 실패({_re}) — '해석중_' 유지")
                    # transient 면 UI 에서 지정한 TavgStart 를 넘겨 시간평균으로 기록.
                    _avg0 = (self.params.get("transient") or {}).get("avg_start")
                    # 사용자가 지정한 밀도를 결과에도 반영한다(종전에는 배치
                    # 경로가 rho 를 넘기지 않아 CSV 에 기본값 1025 가 박혔다).
                    _ex = ResultExtractor(final_dir, speed, angle,
                                          rho=float(self.params.get("rho", 1025.0)))
                    _ex.save_csv(self.output_csv, t_avg_start=_avg0)
                    self.n_success += 1
                    if _ex.is_transient_case():
                        _st = compute_transient_stats(final_dir, t_avg_start=_avg0)
                        _u = _st.get("unsteadiness_Cd")
                        self._log(
                            f"✅ 케이스 완료 [{final_dir.name}] — "
                            f"평균 Cd={_st.get('mean_Cd', float('nan')):.4f} "
                            f"(샘플 {_st.get('n_samples', 0)}개)"
                            + (f" · 변동/평균 {_u*100:.2f}%" if _u is not None else ""))
                        if _u is not None and not _st.get("is_unsteady"):
                            self._log("⚠️ 비정상성 미포착(<1%) — 이 결과를 '정상해석으로 "
                                      "충분하다'는 근거로 쓰지 마세요. 난류모델을 "
                                      "kOmegaSSTDDES 로 바꿔 재검증 권장(보완②)")
                    else:
                        self._log(f"✅ 케이스 완료 [{final_dir.name}] — Cd={_cdv:.4f}")
                else:
                    self.n_failed += 1
                    self._log(f"❌ 케이스 결과 없음 [{running_name}] "
                              f"(forceCoeffs 미생성, 워크플로우 ok={ok})")

            except Exception as e:
                self.n_failed += 1
                self._log(f"❌ 케이스 오류 [{running_name}]: {e}")

            self._progress(round((i + 1) / total * 100, 1), i + 1, total)

        self._log(f"\n✅ 배치 해석 완료: {self.output_csv}")
        return self.results

    def stop(self):
        self._stop_flag.set()

    def _log(self, msg: str):
        if self.log_cb:
            try:
                self.log_cb(msg)
            except Exception:
                pass

    def _progress(self, pct: float, step: int, total: int, label: str = ""):
        if self.progress_cb:
            try:
                self.progress_cb(pct, step, total, label)
            except TypeError:
                # 이전 시그니처(label 없음) 호환
                try:
                    self.progress_cb(pct, step, total)
                except Exception:
                    pass
            except Exception:
                pass
