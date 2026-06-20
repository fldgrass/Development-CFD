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
                 cell_size: float = 0.02,
                 n_cores: Optional[int] = None,
                 nx: int = 1, ny: int = 1):
        self.case_dir   = case_dir
        self.stl_path   = stl_path
        self.speed      = speed
        self.angle_deg  = angle_deg
        self.cell_size  = cell_size
        self.nx         = max(1, int(nx))
        self.ny         = max(1, int(ny))
        self.n_cores    = n_cores or get_cpu_count()
        self.Ux, self.Uy, self.Uz = compute_velocity_vector(speed, angle_deg)
        self.turb       = compute_turbulence_params(speed, length_scale=cell_size)

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
        self._patch_controlDict()
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
        """단위 셀 크기·반복 수에 맞게 blockMeshDict 수정"""
        half_x = self.cell_size * self.nx * 1000 / 2   # mm (X 방향 전체 절반)
        half_y = self.cell_size * self.ny * 1000 / 2   # mm (Y 방향 전체 절반)
        depth  = self.cell_size * 1000 / 2 * 5         # mm (단위 셀 기준 Z 깊이)
        cells_x = max(10, int(20 * self.cell_size * self.nx / 0.02))
        cells_y = max(10, int(20 * self.cell_size * self.ny / 0.02))
        cells_z = max(50, int(100 * self.cell_size / 0.02))

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

    def _patch_controlDict(self):
        """forceCoeffs 기준값 수정"""
        aref = self.cell_size ** 2 * self.nx * self.ny  # 전체 도메인 전면 면적
        ctrl = self.case_dir / "system" / "controlDict"
        # 항력 방향 벡터 (유속 방향)
        drag_dir = f"({self.Ux/self.speed:.4f} 0 {self.Uz/self.speed:.4f})" \
                   if self.speed > 0 else "(1 0 0)"
        replace_in_file(ctrl, {
            "magUInf         1.0;        // 기준 유속 [m/s] - Python에서 교체":
                f"magUInf         {self.speed:.4f};",
            "lRef            0.02;       // 기준 길이 [m] (단위 셀 크기)":
                f"lRef            {self.cell_size:.6f};",
            "Aref            4.0e-4;     // 기준 면적 [m^2] (0.02 x 0.02)":
                f"Aref            {aref:.6e};",
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
                 n_cores: Optional[int] = None):
        self.case_dir      = case_dir
        self.cage_stl      = cage_stl
        self.net_stl       = net_stl
        self.speed         = speed
        self.angle_deg     = angle_deg
        self.cage_D        = cage_diameter
        self.cage_H        = cage_depth
        self.n_cores       = n_cores or get_cpu_count()
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

    def _patch_controlDict(self):
        D, H = self.cage_D, self.cage_H
        aref = D * H
        ctrl = self.case_dir / "system" / "controlDict"
        replace_in_file(ctrl, {
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
                 log_cb: Optional[Callable] = None):
        self.case_dir    = case_dir
        self.n_cores     = n_cores
        self.of_version  = of_version
        self.progress_cb = progress_cb   # progress_cb(percent, step, max_step)
        self.log_cb      = log_cb         # log_cb(line: str)
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
        steps = [
            self.run_blockMesh,
            self.run_surfaceFeatureExtract,
            self.run_snappyHexMesh,
            self.run_solver,
            self.run_reconstructPar,
        ]
        for step_fn in steps:
            if self._stop_flag.is_set():
                return False
            if not step_fn() if step_fn != self.run_solver \
               else step_fn(end_time):
                return False
        return True

    def stop(self):
        """해석 중지"""
        self._stop_flag.set()
        if self._proc:
            self._proc.terminate()

    # ─── 내부 실행 헬퍼 ───────────────────────────────────────────────────

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
                        # 스텝 번호 파싱
                        m = re.match(r"^Time = (\d+)", line)
                        if m:
                            step = int(m.group(1))
                            if self.progress_cb:
                                pct = min(int(step / end_time * 100), 99)
                                self.progress_cb(pct, step, end_time)

            self._proc.wait()
            success = (self._proc.returncode == 0)
            status = "✅ 완료" if success else "❌ 실패"
            self._emit_log(f"\n{status}: {label}")
            return success

        except Exception as e:
            self._emit_log(f"❌ 오류 발생: {e}")
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
        """forces 함수로부터 합력 추출"""
        pp_dir = self.case_dir / "postProcessing"
        force_dirs = list(pp_dir.glob("forces*"))
        for fd in force_dirs:
            time_dirs = sorted(
                [d for d in fd.iterdir() if d.is_dir()],
                key=lambda x: float(x.name) if x.name.replace('.','').isdigit() else 0
            )
            if not time_dirs:
                continue
            f_file = time_dirs[-1] / "forces.dat"
            if not f_file.exists():
                continue
            lines = [l for l in f_file.read_text().splitlines()
                     if not l.startswith("#") and l.strip()]
            if lines:
                parts = lines[-1].split()
                if len(parts) >= 7:
                    try:
                        return {
                            "Fx_N": float(parts[1]),
                            "Fy_N": float(parts[2]),
                            "Fz_N": float(parts[3]),
                        }
                    except (ValueError, IndexError):
                        pass
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

            # 케이스 이름 생성
            case_name = f"{self.mode}_U{speed:.2f}_A{angle:.1f}"
            case_dir  = RESULTS_DIR / self.mode / case_name

            # 케이스 빌드
            try:
                if self.mode == "unit_cell":
                    builder = UnitCellCaseBuilder(
                        case_dir=case_dir,
                        stl_path=self.stl_paths.get("net"),
                        speed=speed, angle_deg=angle,
                        **{k: v for k, v in self.params.items()
                           if k in ["cell_size", "n_cores", "nx", "ny"]}
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

                # 해석 실행
                n_cores = self.params.get("n_cores", get_cpu_count())
                runner = OpenFOAMRunner(
                    case_dir=case_dir,
                    n_cores=n_cores,
                    progress_cb=lambda p, s, e: self._progress(
                        int((i + p/100) / total * 100), s, e
                    ),
                    log_cb=self.log_cb
                )
                ok = runner.run_full_workflow()
                if not ok:
                    raise RuntimeError("워크플로우 실패 — logs/ 폴더 로그 확인")

                # 결과 추출
                extractor = ResultExtractor(case_dir, speed, angle)
                extractor.save_csv(self.output_csv)

            except Exception as e:
                self._log(f"❌ 케이스 오류 [{case_name}]: {e}")

            self._progress(int((i+1) / total * 100), i+1, total)

        self._log(f"\n✅ 배치 해석 완료: {self.output_csv}")
        return self.results

    def stop(self):
        self._stop_flag.set()

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)

    def _progress(self, pct: int, step: int, total: int):
        if self.progress_cb:
            self.progress_cb(pct, step, total)
