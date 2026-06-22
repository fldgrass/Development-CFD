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
from typing import Optional, Dict, List, Tuple, Callable

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
                 solidity: float = None):
        self.case_dir         = case_dir
        self.stl_path         = stl_path
        self.speed            = speed
        self.angle_deg        = angle_deg
        self.nx               = max(1, int(nx))
        self.ny               = max(1, int(ny))
        self.n_cores          = n_cores or get_cpu_count()
        self.residual_control = max(1e-5, min(1e-3, float(residual_control)))
        self.end_time         = max(100, int(end_time))
        self.write_interval   = max(10, int(write_interval))
        self.refine_level     = max(1, min(4, int(refine_level)))

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

        bmd = self.case_dir / "system" / "blockMeshDict"
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

        if _cell_proj > 0:
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
                 write_interval: int = 100):
        self.case_dir         = case_dir
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
        if self.cage_stl and self.cage_stl.exists():
            shutil.copy2(self.cage_stl, trisurf_dir / "cageSurface.stl")
        if self.net_stl and self.net_stl.exists():
            shutil.copy2(self.net_stl, trisurf_dir / "netSurface.stl")

        self._patch_velocity_fields()
        self._patch_turbulence_fields()
        self._patch_blockMesh()
        self._patch_fvSolution()
        self._patch_snappyHexMesh()
        self._patch_controlDict()
        self._patch_decomposePar()

        logger.info(f"[FullStructure] 케이스 생성 완료: {self.case_dir}")
        return self.case_dir

    def _patch_velocity_fields(self):
        Uvec = f"({self.Ux:.6f} {self.Uy:.6f} {self.Uz:.6f})"
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

    def _patch_blockMesh(self):
        """도메인 크기를 가두리 크기에 맞게 자동 계산"""
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
        """가두리 크기에 맞게 정밀화 박스 조정"""
        D, H = self.cage_D, self.cage_H
        r = D / 2 * 1.2
        snappy = self.case_dir / "system" / "snappyHexMeshDict"
        replace_in_file(snappy, {
            "min     (-6 -6 -6);": f"min     ({-r:.2f} {-r:.2f} {-(H+1):.2f});",
            "max     ( 6  6  1);": f"max     ({r:.2f}  {r:.2f}  1.0);",
            "locationInMesh (0 0 -2.5);": f"locationInMesh (0 0 {-H/2:.2f});",
        })

    def _patch_fvSolution(self):
        """수렴 기준(residualControl) 주입"""
        fvs = self.case_dir / "system" / "fvSolution"
        replace_in_file(fvs, {
            "residualLevel   1e-4;": f"residualLevel   {self.residual_control:.0e};",
        })

    def _patch_controlDict(self):
        D, H = self.cage_D, self.cage_H
        aref = D * H
        ctrl = self.case_dir / "system" / "controlDict"
        replace_in_file(ctrl, {
            "endTimeValue        3000;":       f"endTimeValue        {self.end_time};",
            "writeIntervalValue  100;":        f"writeIntervalValue  {self.write_interval};",
            "magUInf         1.0;": f"magUInf         {self.speed:.4f};",
            "lRef            10.0;":  f"lRef            {D:.4f};",
            "Aref            50.0;":  f"Aref            {aref:.4f};",
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
        if self._has_cyclic_patches():
            # OpenFOAM v2312 버그: 병렬 snappyHexMesh + cyclic 패치 →
            # globalIndexAndTransform 충돌(transform sign mismatch) → serial로 우회
            return self._run_step(
                "snappyHexMesh -overwrite", "격자 스냅 (snappyHexMesh, 직렬)")
        cmd = f"mpirun --oversubscribe -np {self.n_cores} snappyHexMesh -overwrite -parallel"
        return self._run_step(cmd, "격자 스냅 (snappyHexMesh)", parallel=True,
                              pre_cmd="decomposePar -force")

    def run_decomposePar(self) -> bool:
        return self._run_step("decomposePar -force", "도메인 분할 (decomposePar)")

    def run_solver(self, end_time: int = 2000) -> bool:
        """병렬 simpleFoam 실행 + 실시간 잔차 모니터링"""
        cmd = f"mpirun --oversubscribe -np {self.n_cores} simpleFoam -parallel"
        # cyclic 케이스는 serial snappyHexMesh 후 분할이 안 됐으므로 여기서 decomposePar 실행
        pre = "decomposePar -force" if self._has_cyclic_patches() else None
        return self._run_step(cmd, "CFD 해석 (simpleFoam)", parallel=True,
                              pre_cmd=pre,
                              monitor_residuals=True, end_time=end_time)

    def run_reconstructPar(self) -> bool:
        return self._run_step("reconstructPar -latestTime", "결과 재조합 (reconstructPar)")

    # ─── 전체 워크플로우 실행 ─────────────────────────────────────────────

    def run_full_workflow(self, end_time: int = 2000) -> bool:
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
        if not self.run_solver(end_time):
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
                  end_time: int = 2000) -> bool:
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
                        m = re.match(r"^Time = (\d+)", line)
                        if m:
                            step = int(m.group(1))
                            pct = round(min(step / end_time * 100, 99.9), 1)
                            residual_info = ""
                            if self.last_residuals:
                                max_r = max(self.last_residuals.values())
                                residual_info = f"잔차 {max_r:.2e}"
                            self._emit_step(label, "running", pct,
                                            f"Time={step}/{end_time}  {residual_info}")
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

    def save_csv(self, output_path: Path) -> Path:
        """결과를 CSV 파일로 저장 (질량-스프링 모델 호환 포맷)"""
        coeffs = self.extract_force_coeffs() or {}
        forces = self.extract_forces() or {}

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
        file_exists = output_path.exists()

        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"결과 저장: {output_path}")
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
                 log_cb: Optional[Callable] = None):
        self.mode        = mode          # "unit_cell" or "full_structure"
        self.stl_paths   = stl_paths
        self.speeds      = speeds
        self.angles      = angles
        self.output_csv  = output_csv
        self.params      = common_params or {}
        self.progress_cb = progress_cb
        self.log_cb      = log_cb
        self._stop_flag  = threading.Event()
        self.results: List[Dict] = []

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
            case_dir  = RESULTS_DIR / self.mode / running_name
            done_dir  = RESULTS_DIR / self.mode / done_name
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
                                    "write_interval", "refine_level", "solidity"]}
                    )
                else:
                    builder = FullStructureCaseBuilder(
                        case_dir=case_dir,
                        cage_stl=self.stl_paths.get("cage"),
                        net_stl=self.stl_paths.get("net"),
                        speed=speed, angle_deg=angle,
                        **{k: v for k, v in self.params.items()
                           if k in ["cage_diameter", "cage_depth", "n_cores"]}
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
                    ResultExtractor(final_dir, speed, angle).save_csv(self.output_csv)
                    self._log(f"✅ 케이스 완료 [{final_dir.name}] — Cd={_cdv:.4f}")
                else:
                    self._log(f"❌ 케이스 결과 없음 [{running_name}] "
                              f"(forceCoeffs 미생성, 워크플로우 ok={ok})")

            except Exception as e:
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
