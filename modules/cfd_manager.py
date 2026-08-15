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
        t, ok = _sub_once(
            t, r"(purgeWrite\s+\d+;)",
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
                 aref_override: Optional[float] = None):
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
        self.refine_level     = max(1, min(6, int(refine_level)))

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
        """격자 세분화 레벨 주입"""
        level_min = max(1, self.refine_level - 1)
        level_max = self.refine_level
        snappy = self.case_dir / "system" / "snappyHexMeshDict"
        replace_in_file(snappy, {
            "refineLevelMin  2;": f"refineLevelMin  {level_min};",
            "refineLevelMax  3;": f"refineLevelMax  {level_max};",
        })

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
                 n_layers: int = 0):
        # 격자 옵션(보완④: DDES 등에서 격자 민감도를 확인하기 위한 노브).
        # 기본값 refine_level=3 / n_layers=0 은 종전 하드코딩 값과 완전히 동일한
        # snappyHexMeshDict 를 만든다(회귀 방지).
        self.refine_level     = max(1, min(5, int(refine_level)))
        self.n_layers         = max(0, min(10, int(n_layers)))
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

        self._patch_velocity_fields()
        self._patch_turbulence_fields()
        self._patch_blockMesh()
        self._patch_fvSolution()
        self._patch_snappyHexMesh()
        self._patch_controlDict()
        self._patch_decomposePar()

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
        # 도메인: 상류 3L, 하류 7L, 횡·수직 ±3L (net 이 충분히 도메인 안에 들도록)
        self._dom_min = (cx-3*L, cy-3*L, cz-3*L)
        self._dom_max = (cx+7*L, cy+3*L, cz+3*L)
        # 정밀화 박스: net + 근접 후류
        self._box_min = (bxmin-0.5*L, bymin-0.5*L, bzmin-0.5*L)
        self._box_max = (bxmax+1.5*L, bymax+0.5*L, bzmax+0.5*L)
        # 기준점: net 상류(연결된 유체 영역 어디든 가능, net 표면만 피하면 됨)
        self._loc = (cx-2.5*L, cy, cz)

    def _patch_blockMesh(self):
        """도메인 크기 자동 계산. net-only 면 net 크기 기준, 아니면 가두리 크기 기준."""
        if getattr(self, "_net_only", False):
            x_min, y_min, z_min = self._dom_min
            x_max, y_max, z_max = self._dom_max
            L = self._net_L
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
            "(-30  -30  -5)": f"({x_min:.1f}  {y_min:.1f}  {z_min:.1f})",
            "( 70  -30  -5)": f"({x_max:.1f}  {y_min:.1f}  {z_min:.1f})",
            "( 70   30  -5)": f"({x_max:.1f}  {y_max:.1f}  {z_min:.1f})",
            "(-30   30  -5)": f"({x_min:.1f}  {y_max:.1f}  {z_min:.1f})",
            "(-30  -30   0)": f"({x_min:.1f}  {y_min:.1f}  {z_max:.1f})",
            "( 70  -30   0)": f"({x_max:.1f}  {y_min:.1f}  {z_max:.1f})",
            "( 70   30   0)": f"({x_max:.1f}  {y_max:.1f}  {z_max:.1f})",
            "(-30   30   0)": f"({x_min:.1f}  {y_max:.1f}  {z_max:.1f})",
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
        refineBox {{ mode inside; levels ((1e10 {_lmin})); }}
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
            # 최신 시간을 그대로 이어받는다(startFrom latestTime).
            t0 = self._latest_processor_time()
            self._emit_log(f"선행 수렴 완료 — t={t0:g} 에서 비정상 해석 이어받기")
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
        # 이어받기면 latestTime 에서 시작
        ctrl = self.case_dir / "system" / "controlDict"
        c = ctrl.read_text()
        c = re.sub(r"startFrom\s+\w+;",
                   f"startFrom       {'latestTime' if init else 'startTime'};", c)
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

        self._emit_log(f"2단계: pimpleFoam 비정상 해석 (t={t0:g} → {end_abs:g} s)")
        # 이어받기면 이미 분할된 processor 결과를 써야 하므로 재분할 금지
        cmd = f"mpirun --oversubscribe -np {self.n_cores} pimpleFoam -parallel"
        return self._run_step(cmd, "CFD 해석 (pimpleFoam)", parallel=True,
                              pre_cmd=None if init else "decomposePar -force",
                              monitor_residuals=True, end_time=end_abs,
                              start_time=t0)

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
        "rho_kg_m3", "case_name", "timestamp"
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
                                    "aref_override"]}
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
                                    "aref_override", "refine_level", "n_layers"]}
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
                    _ex = ResultExtractor(final_dir, speed, angle)
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
