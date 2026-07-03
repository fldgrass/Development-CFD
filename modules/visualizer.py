"""
visualizer.py
=============
PyVista 기반 3D CFD 결과 실시간 시각화 모듈
- OpenFOAM 결과 읽기 (VTK/OpenFOAM 포맷)
- 유속장, 압력장, 난류 운동에너지 렌더링
- Streamlit 임베드 가능한 HTML/PNG 출력
- 10초 주기 자동 갱신
"""

import os
import re
import time
import math
import logging
import tempfile
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger("visualizer")
logger.setLevel(logging.INFO)

# ─── PyVista 임포트 (없으면 graceful fallback) ────────────────────────────
try:
    import pyvista as pv
    import vtk
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"
    PYVISTA_OK = True
    logger.info("PyVista 초기화 성공")
except ImportError:
    PYVISTA_OK = False
    logger.warning("PyVista 미설치 → 모의 시각화 모드 사용")

# ─── Matplotlib fallback ──────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# ═══════════════════════════════════════════════════════════════════════════
# OpenFOAM 결과 파서
# ═══════════════════════════════════════════════════════════════════════════

class OpenFOAMResultReader:
    """
    OpenFOAM 케이스 결과 디렉토리에서 최신 시간 스텝 자동 로드
    """

    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)

    def get_time_steps(self) -> List[float]:
        """수치 이름의 시간 디렉토리 목록 반환 (정렬)"""
        steps = []
        for d in self.case_dir.iterdir():
            if d.is_dir():
                try:
                    t = float(d.name)
                    if t > 0:
                        steps.append(t)
                except ValueError:
                    pass
        return sorted(steps)

    def get_latest_time(self) -> Optional[float]:
        steps = self.get_time_steps()
        return steps[-1] if steps else None

    def load_openfoam_mesh(self) -> Optional[Any]:
        """PyVista로 OpenFOAM 케이스 로드"""
        if not PYVISTA_OK:
            return None
        try:
            # OpenFOAM 케이스 파일 (.foam) 자동 생성
            foam_file = self.case_dir / f"{self.case_dir.name}.foam"
            if not foam_file.exists():
                foam_file.write_text("")

            reader = pv.OpenFOAMReader(str(foam_file))
            reader.set_active_time_value(reader.time_values[-1]
                                         if reader.time_values else 0)
            mesh = reader.read()
            return mesh
        except Exception as e:
            logger.warning(f"OpenFOAM 메시 로드 실패: {e}")
            return None

    def read_sampled_data(self, field: str = "U") -> Optional[Dict]:
        """postProcessing 샘플 데이터 읽기"""
        pp_dir = self.case_dir / "postProcessing" / "velocitySampling"
        if not pp_dir.exists():
            return None

        time_dirs = sorted(
            [d for d in pp_dir.iterdir() if d.is_dir()],
            key=lambda x: float(x.name) if x.name.replace('.','',1).isdigit() else 0
        )
        if not time_dirs:
            return None

        latest = time_dirs[-1]
        results = {}
        for csv_file in latest.glob("*.csv"):
            try:
                data = np.genfromtxt(csv_file, delimiter=",",
                                     skip_header=1, filling_values=np.nan)
                results[csv_file.stem] = data
            except Exception as e:
                logger.warning(f"CSV 읽기 실패 {csv_file}: {e}")
        return results if results else None

    def read_residuals(self) -> Optional[Dict[str, List[float]]]:
        """로그 파일에서 잔차 히스토리 파싱.
        케이스 폴더 내부 + 프로젝트 루트 logs/ 폴더를 함께 탐색."""
        from pathlib import Path
        import re

        # 케이스 이름 패턴으로 전역 logs/ 폴더도 탐색
        _global_logs = Path(__file__).resolve().parent.parent / "logs"
        log_files = (
            list(self.case_dir.glob("*.log")) +
            list(self.case_dir.glob("log.*")) +
            list(_global_logs.glob(f"{self.case_dir.name}*.log"))
            if _global_logs.exists() else
            list(self.case_dir.glob("*.log")) + list(self.case_dir.glob("log.*"))
        )
        if not log_files:
            return None

        # simpleFoam 로그만 우선 (가장 크고 최신인 것)
        log_file = max(log_files, key=lambda f: f.stat().st_size)
        residuals: Dict[str, List[float]] = {}

        pattern = re.compile(
            r"Solving for (\w+),\s+Initial residual = ([\d.e+-]+)"
        )
        try:
            for line in log_file.read_text(errors="replace").splitlines():
                m = pattern.search(line)
                if m:
                    field = m.group(1)
                    val   = float(m.group(2))
                    residuals.setdefault(field, []).append(val)
        except Exception:
            pass

        return residuals if residuals else None


# ═══════════════════════════════════════════════════════════════════════════
# 3D 시각화 렌더러
# ═══════════════════════════════════════════════════════════════════════════

class CFDVisualizer:
    """
    CFD 결과 3D 시각화 (PyVista 기반)
    Streamlit에 PNG/HTML로 임베드
    """

    # 컬러맵 설정
    FIELD_CONFIG = {
        "U":     {"label": "유속 |U| [m/s]",       "cmap": "coolwarm",  "component": "magnitude"},
        "p":     {"label": "압력 p [Pa·m²/s²]",    "cmap": "RdBu_r",    "component": "scalar"},
        "k":     {"label": "난류 운동에너지 k",     "cmap": "hot_r",     "component": "scalar"},
        "omega": {"label": "비소산율 ω",            "cmap": "plasma",    "component": "scalar"},
        "nut":   {"label": "난류 점성계수 νt",      "cmap": "viridis",   "component": "scalar"},
    }

    def __init__(self, case_dir: Path, window_size: Tuple[int,int] = (900, 600),
                 mode: Optional[str] = None, angle_deg: Optional[float] = None):
        self.case_dir    = Path(case_dir)
        self.window_size = window_size
        self.reader      = OpenFOAMResultReader(case_dir)
        self._mesh_cache = None
        self._cache_time = 0
        # 통일 표시 좌표계: 솔버→표시 회전행렬 R(모드/영각 자동 감지, 미검출 시 변환 생략).
        # 메시·STL·벡터장을 R 로 회전해 두 모드를 동일 프레임(그물 Y-Z, 유속 X-Y)으로 표시.
        self.mode, self.angle_deg = self._detect_mode_angle(mode, angle_deg)
        self._disp_R = self._compute_display_R()
        self._disp_M = None        # 4×4(중심 기준 회전), _apply_display_frame 에서 설정
        self._disp_center = None

    def _detect_mode_angle(self, mode, angle_deg):
        """케이스 경로/이름에서 모드와 영각을 감지(인자가 주어지면 우선)."""
        s = str(self.case_dir).lower()
        if mode is None:
            if "full_structure" in s:
                mode = "full_structure"
            elif "unit_cell" in s:
                mode = "unit_cell"
        if angle_deg is None:
            m = re.search(r"_a(\d+(?:\.\d+)?)", self.case_dir.name.lower())
            if m:
                try:
                    angle_deg = float(m.group(1))
                except ValueError:
                    angle_deg = None
        return mode, angle_deg

    def _compute_display_R(self):
        """모드/영각 기반 솔버→표시 회전행렬(3×3). 미검출 시 None(변환 생략)."""
        if self.mode not in ("unit_cell", "full_structure"):
            return None
        ang = self.angle_deg if self.angle_deg is not None else 0.0
        try:
            from cfd_manager import solver_to_display_rotation
            return solver_to_display_rotation(self.mode, ang)
        except Exception:
            return None

    def _mesh_center(self, mesh):
        try:
            internal = (mesh["internalMesh"] if mesh is not None
                        and "internalMesh" in mesh.keys() else None)
            b = internal.bounds if internal is not None else mesh.bounds
            return ((b[0]+b[1])/2.0, (b[2]+b[3])/2.0, (b[4]+b[5])/2.0)
        except Exception:
            return (0.0, 0.0, 0.0)

    def _apply_display_frame(self, mesh):
        """로드한 메시(MultiBlock)를 통일 표시 프레임으로 회전(중심 기준, 벡터장 포함).
        실패 시 원본을 그대로 반환(안전 강등)."""
        if self._disp_R is None or mesh is None:
            return mesh
        try:
            c = np.asarray(self._mesh_center(mesh), float)
            M = np.eye(4)
            M[:3, :3] = np.asarray(self._disp_R, float)
            M[:3, 3] = c - M[:3, :3] @ c       # 중심 기준 회전
            self._disp_M = M
            self._disp_center = c
            for key in list(mesh.keys()):
                blk = mesh[key]
                if blk is None:
                    continue
                try:
                    mesh[key] = blk.transform(
                        M, transform_all_input_vectors=True, inplace=False)
                except TypeError:
                    mesh[key] = blk.transform(M, inplace=False)
        except Exception as _e:
            logger.warning(f"표시 프레임 변환 실패(원본 사용): {_e}")
        return mesh

    # ─── 메인 렌더 함수 ───────────────────────────────────────────────────

    def render_field(self, field: str = "U",
                     slice_normal: str = "y",
                     slice_origin: Optional[Tuple] = None,
                     show_streamlines: bool = False,
                     show_surface: bool = True) -> Optional[str]:
        """
        유동장 시각화 → PNG 파일 경로 반환
        Args:
            field: 시각화할 장 ("U", "p", "k", ...)
            slice_normal: 슬라이스 법선 방향
            slice_origin: 슬라이스 중심 좌표 (None이면 자동)
            show_streamlines: 유선(streamlines) 표시 여부
        Returns:
            저장된 PNG 파일 경로
        """
        if PYVISTA_OK:
            return self._render_pyvista(field, slice_normal, slice_origin,
                                        show_streamlines, show_surface)
        elif MATPLOTLIB_OK:
            return self._render_matplotlib_fallback(field)
        else:
            return self._render_placeholder(field)

    def _render_pyvista(self, field: str, slice_normal: str,
                        slice_origin: Optional[Tuple],
                        show_streamlines: bool,
                        show_surface: bool) -> Optional[str]:
        """PyVista 3D 렌더링"""
        try:
            mesh = self._get_mesh()
            if mesh is None:
                return self._render_placeholder(field)

            cfg = self.FIELD_CONFIG.get(field, self.FIELD_CONFIG["U"])
            plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
            plotter.set_background("white")

            # 내부 메시 추출
            internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() \
                       else mesh

            # 슬라이스 생성
            normal_map = {"x": (1,0,0), "y": (0,1,0), "z": (0,0,1)}
            normal_vec = normal_map.get(slice_normal, (0,1,0))

            bounds = internal.bounds
            if slice_origin is None:
                cx = (bounds[0] + bounds[1]) / 2
                cy = (bounds[2] + bounds[3]) / 2
                cz = (bounds[4] + bounds[5]) / 2
                slice_origin = (cx, cy, cz)

            sliced = internal.slice(normal=normal_vec, origin=slice_origin)

            # 필드 데이터 매핑
            if field in sliced.array_names:
                arr = sliced[field]
                if arr.ndim == 2:  # 벡터 → 크기
                    scalar = np.linalg.norm(arr, axis=1)
                    sliced["display_field"] = scalar
                    display_field = "display_field"
                else:
                    display_field = field

                plotter.add_mesh(
                    sliced, scalars=display_field,
                    cmap=cfg["cmap"],
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "title": cfg["label"],
                        "title_font_size": 14,
                        "label_font_size": 11,
                        "n_labels": 5,
                        "italic": False,
                        "fmt": "%.3f",
                        "position_x": 0.85,
                        "position_y": 0.05,
                    }
                )
            else:
                plotter.add_mesh(sliced, color="lightblue")

            # 경계면 표시
            if show_surface:
                for key in mesh.keys():
                    if key != "internalMesh":
                        try:
                            patch = mesh[key]
                            plotter.add_mesh(
                                patch, color="gray",
                                opacity=0.3,
                                show_edges=False
                            )
                        except Exception:
                            pass

            # 유선 표시
            if show_streamlines and "U" in internal.array_names:
                try:
                    seeds = pv.Sphere(radius=0.1, center=slice_origin)
                    stream = internal.streamlines_from_source(
                        seeds, vectors="U",
                        max_steps=1000,
                        max_step_length=0.1,
                    )
                    plotter.add_mesh(stream, color="black", line_width=1)
                except Exception as e:
                    logger.warning(f"유선 렌더링 실패: {e}")

            # 카메라 설정
            plotter.view_isometric()
            plotter.add_axes()
            plotter.add_title(
                f"{field} 유동장 시각화 | {self.case_dir.name}",
                font_size=12
            )

            # 저장
            out_path = self._get_output_path(f"{field}_{slice_normal}")
            plotter.screenshot(out_path, transparent_background=False)
            plotter.close()

            logger.info(f"렌더링 완료: {out_path}")
            return str(out_path)

        except Exception as e:
            logger.error(f"PyVista 렌더링 오류: {e}")
            return self._render_placeholder(field)

    def _render_matplotlib_fallback(self, field: str) -> str:
        """PyVista 없을 때 Matplotlib 2D 대체 렌더링"""
        cfg = self.FIELD_CONFIG.get(field, self.FIELD_CONFIG["U"])

        # 잔차 데이터 읽기
        residuals = self.reader.read_residuals()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"CFD 해석 결과 모니터 | {self.case_dir.name}", fontsize=13)

        # 잔차 수렴 그래프
        ax1 = axes[0]
        if residuals:
            for fname, vals in residuals.items():
                if vals:
                    ax1.semilogy(vals, label=fname, linewidth=1.5)
            ax1.set_xlabel("반복 횟수 (Iteration)", fontsize=11)
            ax1.set_ylabel("잔차 (Residual)", fontsize=11)
            ax1.set_title("수렴 이력", fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1e-4, color='red', linestyle='--',
                        alpha=0.7, label='수렴 기준 (1e-4)')
        else:
            ax1.text(0.5, 0.5, "해석 데이터 없음\n또는 해석 진행 중",
                     ha='center', va='center', fontsize=12,
                     transform=ax1.transAxes)
            ax1.set_title("수렴 이력 (대기 중...)", fontsize=12)

        # 샘플 유속 분포 (postProcessing CSV)
        ax2 = axes[1]
        sampled = self.reader.read_sampled_data()
        if sampled:
            for name, data in sampled.items():
                if data is not None and data.ndim == 2 and data.shape[1] >= 4:
                    x_pos = data[:, 0]
                    # U 크기 계산
                    try:
                        u_mag = np.sqrt(
                            data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2
                        )
                        ax2.plot(x_pos, u_mag, label=name, linewidth=1.5)
                    except Exception:
                        pass
            ax2.set_xlabel("위치 [m]", fontsize=11)
            ax2.set_ylabel("유속 |U| [m/s]", fontsize=11)
            ax2.set_title("유속 분포 (중심선)", fontsize=12)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "샘플링 데이터 없음\n(해석 완료 후 표시)",
                     ha='center', va='center', fontsize=12,
                     transform=ax2.transAxes)
            ax2.set_title(f"{cfg['label']} 분포 (대기 중...)", fontsize=12)

        plt.tight_layout()
        out_path = self._get_output_path(f"{field}_2d")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        return str(out_path)

    def _render_placeholder(self, field: str) -> str:
        """시각화 라이브러리 없을 때 플레이스홀더 이미지"""
        if not MATPLOTLIB_OK:
            return ""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.55, "🌊 CFD 해석 준비 중...",
                ha='center', va='center', fontsize=16,
                transform=ax.transAxes, color='steelblue')
        ax.text(0.5, 0.40,
                f"케이스: {self.case_dir.name}\n필드: {field}",
                ha='center', va='center', fontsize=11,
                transform=ax.transAxes, color='gray')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#f0f8ff")
        out_path = self._get_output_path(f"placeholder_{field}")
        plt.savefig(out_path, dpi=100)
        plt.close()
        return str(out_path)

    # ─── 잔차 수렴 그래프 ────────────────────────────────────────────────

    def plot_residuals(self) -> Optional[str]:
        """잔차 수렴 이력 그래프 생성"""
        if not MATPLOTLIB_OK:
            return None

        residuals = self.reader.read_residuals()
        if not residuals:
            return self._render_placeholder("residuals")

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, (field, vals) in enumerate(residuals.items()):
            if vals:
                color = colors[i % len(colors)]
                ax.semilogy(vals, label=field,
                            color=color, linewidth=1.8)

        ax.axhline(y=1e-4, color='red', linestyle='--',
                   alpha=0.6, linewidth=1, label='수렴 기준 (1e-4)')
        ax.set_xlabel("반복 횟수", fontsize=12)
        ax.set_ylabel("잔차 (log scale)", fontsize=12)
        ax.set_title("수렴 이력 모니터", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_facecolor("#fafafa")
        plt.tight_layout()

        out_path = self._get_output_path("residuals")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        return str(out_path)

    # ─── 유속 감쇠 프로파일 ──────────────────────────────────────────────

    def plot_velocity_attenuation(self, u_inlet: float = 1.0) -> Optional[str]:
        """가두리 전후 유속 감쇠 프로파일 시각화"""
        if not MATPLOTLIB_OK:
            return None

        sampled = self.reader.read_sampled_data()
        if not sampled:
            return self._render_placeholder("velocity")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("가두리 유속 감쇠 분석", fontsize=13, fontweight='bold')

        ax1 = axes[0]
        ax2 = axes[1]

        for name, data in sampled.items():
            if data is None or data.ndim < 2 or data.shape[0] < 2:
                continue
            try:
                if data.shape[1] >= 4:
                    pos = data[:, 0]
                    u_mag = np.sqrt(data[:,1]**2 + data[:,2]**2 + data[:,3]**2)
                    ax1.plot(pos, u_mag, 'o-', label=name,
                             linewidth=1.5, markersize=3)
                    ax2.plot(pos, u_mag / u_inlet * 100, 's-',
                             label=f"{name} (%)", linewidth=1.5, markersize=3)
            except Exception:
                pass

        ax1.set_xlabel("위치 [m]", fontsize=11)
        ax1.set_ylabel("유속 |U| [m/s]", fontsize=11)
        ax1.set_title("유속 분포 (절대값)", fontsize=12)
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("위치 [m]", fontsize=11)
        ax2.set_ylabel("유속 감쇠율 [%]", fontsize=11)
        ax2.set_title("유속 감쇠율 (입구 대비)", fontsize=12)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        out_path = self._get_output_path("velocity_attenuation")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        return str(out_path)

    # ─── 유체력 계수 분포 ────────────────────────────────────────────────

    def plot_force_coefficients(self, csv_path: Path) -> Optional[str]:
        """Cd/Cl 계수 분포도 (영각/유속별)"""
        if not MATPLOTLIB_OK or not csv_path.exists():
            return None

        try:
            import csv as csvmod
            rows = []
            with open(csv_path) as f:
                reader = csvmod.DictReader(f)
                for row in reader:
                    rows.append(row)

            if not rows:
                return None

            angles = sorted(set(float(r["angle_deg"]) for r in rows))
            speeds = sorted(set(float(r["speed_m_s"]) for r in rows))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("유체력 계수 데이터베이스", fontsize=13, fontweight='bold')

            colors = plt.cm.viridis(np.linspace(0, 1, len(speeds)))

            for ax_idx, coeff in enumerate(["Cd", "Cl"]):
                ax = axes[ax_idx]
                for j, speed in enumerate(speeds):
                    speed_rows = [r for r in rows
                                  if abs(float(r["speed_m_s"]) - speed) < 0.01]
                    if not speed_rows:
                        continue
                    ang_arr = [float(r["angle_deg"]) for r in speed_rows]
                    val_arr = [float(r[coeff]) for r in speed_rows
                               if r[coeff] not in ("nan", "")]
                    if val_arr:
                        ax.plot(ang_arr[:len(val_arr)], val_arr,
                                'o-', color=colors[j],
                                label=f"U={speed:.2f} m/s",
                                linewidth=1.8, markersize=5)

                ax.set_xlabel("영각 [°]", fontsize=11)
                ax.set_ylabel(coeff, fontsize=11)
                ax.set_title(f"{coeff} vs 영각", fontsize=12)
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out_path = self._get_output_path("force_coeffs")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            return str(out_path)

        except Exception as e:
            logger.error(f"계수 플롯 오류: {e}")
            return None

    # ─── 유틸리티 ─────────────────────────────────────────────────────────

    def _get_mesh(self, max_cache_age: float = 30.0):
        """메시 캐시 관리 (30초 캐시)"""
        now = time.time()
        if self._mesh_cache is None or (now - self._cache_time) > max_cache_age:
            _m = self.reader.load_openfoam_mesh()
            self._mesh_cache = self._apply_display_frame(_m)
            self._cache_time = now
        return self._mesh_cache

    def _get_output_path(self, suffix: str) -> Path:
        out_dir = Path(tempfile.gettempdir()) / "cfd_viz"
        out_dir.mkdir(exist_ok=True)
        return out_dir / f"{self.case_dir.name}_{suffix}.png"

    def invalidate_cache(self):
        """캐시 무효화 (새 결과가 생성된 경우 호출)"""
        self._mesh_cache = None
        self._cache_time = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Plotly 인터랙티브 시각화 (마우스 드래그 회전, 슬라이스 위치 이동)
    # ═══════════════════════════════════════════════════════════════════════

    # 필드별 Plotly 컬러스케일 매핑
    _PLOTLY_CMAP = {
        "U":     "Jet",
        "p":     "RdBu_r",
        "k":     "Hot",
        "omega": "Plasma",
        "nut":   "Viridis",
    }
    _FIELD_UNIT = {
        "U": "m/s", "p": "Pa·m²/s²", "k": "m²/s²",
        "omega": "1/s", "nut": "m²/s",
    }

    def render_field_plotly(self,
                             field: str = "U",
                             slice_normal: str = "y",
                             slice_fraction: float = 0.5,
                             show_streamlines: bool = False,
                             tile_nx: int = 1,
                             tile_ny: int = 1,
                             init_camera: bool = True,
                             n_frames: int = 1,
                             stl_opacity: float = 0.15) -> Optional[Any]:
        """
        PyVista로 슬라이스 추출 → Plotly go.Mesh3d 인터랙티브 3D 뷰어 반환.
        n_frames > 1: Streamlit 슬라이더 대신 Plotly 내장 슬라이더로 위치 제어.
        Plotly 내장 슬라이더는 Streamlit 리런을 유발하지 않으므로 카메라가 유지된다.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not PYVISTA_OK:
            return None

        try:
            mesh = self._get_mesh()
            if mesh is None:
                return None

            internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh

            # 슬라이스 위치 계산
            bounds = internal.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
            axis_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
            b_lo, b_hi = axis_map.get(slice_normal, (2, 3))
            lo, hi = bounds[b_lo], bounds[b_hi]
            cx = (bounds[0]+bounds[1])/2
            cy = (bounds[2]+bounds[3])/2
            cz = (bounds[4]+bounds[5])/2

            # ── v12 항목2·3: 평면 격자 샘플링 기반 고해상 슬라이스 ──────────
            # 종전(메시 단면 추출)은 해상도가 국소 셀 크기(원방 2.5~15mm)에 묶여
            # 원방에서 블록형으로 보였다. 물체 중심 뷰 박스에 N×N 균일 격자를
            # 깔고 내부 메시를 직접 프로브(선형 보간) → 셀 크기와 무관한 매끈한
            # 단면(go.Surface, 4배 이상 세밀). 전 프레임 점을 단일 PolyData 로
            # 묶어 1회 배치 프로브(locator 1회 구축)해 상호작용 성능을 유지한다.
            _vc, _vh = self._view_box(bounds)
            _axmap = {"x": 0, "y": 1, "z": 2}
            n_ax = _axmap.get(slice_normal, 1)
            in_ax = [a for a in range(3) if a != n_ax]
            _N = 160
            g1 = np.linspace(_vc[in_ax[0]]-_vh, _vc[in_ax[0]]+_vh, _N)
            g2 = np.linspace(_vc[in_ax[1]]-_vh, _vc[in_ax[1]]+_vh, _N)
            G1, G2 = np.meshgrid(g1, g2, indexing="ij")
            if n_frames > 1:
                _fracs = self._adaptive_fracs(mesh, slice_normal, bounds, n_frames)
            else:
                _fracs = np.array([min(max(float(slice_fraction), 0.0), 1.0)])
            _npts = _N * _N
            _allpts = np.empty((len(_fracs) * _npts, 3))
            for _fi, _fr in enumerate(_fracs):
                _blk = _allpts[_fi*_npts:(_fi+1)*_npts]
                _blk[:, n_ax] = lo + (hi - lo) * float(_fr)
                _blk[:, in_ax[0]] = G1.ravel()
                _blk[:, in_ax[1]] = G2.ravel()
            _sam = pv.PolyData(_allpts).sample(internal)
            if field not in _sam.array_names:
                return None
            _arr = np.asarray(_sam[field])
            _vals_all = (np.linalg.norm(_arr, axis=1)
                         if _arr.ndim == 2 else _arr.astype(float))
            if "vtkValidPointMask" in _sam.array_names:
                _vmk = np.asarray(_sam["vtkValidPointMask"])
                _vals_all = np.where(_vmk > 0, _vals_all, np.nan)
            _fin = _vals_all[np.isfinite(_vals_all)]
            if _fin.size == 0:
                return None
            # 컬러 범위: 전 프레임 공통(속도류는 0~최대 — v11 항목5와 일관)
            _cmax = float(_fin.max())
            _cmin = 0.0 if float(_fin.min()) >= 0.0 else float(_fin.min())
            if not (_cmax > _cmin):
                _cmax = _cmin + 1e-9
            _frame_vals = {round(float(_fr), 6):
                           _vals_all[_fi*_npts:(_fi+1)*_npts]
                           .reshape(_N, _N).astype(np.float32)
                           for _fi, _fr in enumerate(_fracs)}
            _G1f, _G2f = G1.astype(np.float32), G2.astype(np.float32)

            cmap   = self._PLOTLY_CMAP.get(field, "Jet")
            unit   = self._FIELD_UNIT.get(field, "")

            # ── 타일링 오프셋 (주기 단위셀 nx×ny 복제 — 시각화 전용) ────────
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            tile_nx = max(1, int(tile_nx))
            tile_ny = max(1, int(tile_ny))
            _offsets = [(i * dx, j * dy)
                        for i in range(tile_nx) for j in range(tile_ny)]

            # 경계면 패치 미리 추출(타일마다 복제)
            _patches = []
            for key in mesh.keys():
                if key == "internalMesh":
                    continue
                try:
                    patch = mesh[key]
                    if patch.n_points == 0:
                        continue
                    ptri = patch.triangulate()
                    _patches.append((ptri.points,
                                     ptri.faces.reshape(-1, 4)[:, 1:]))
                except Exception:
                    pass

            # ── 슬라이스 트레이스 생성 클로저 (배치 프로브 결과 → go.Surface) ──
            def _make_slice_traces(frac):
                """frac 위치의 고해상 슬라이스 Surface 리스트(타일별 1개) 반환."""
                frac = float(frac)
                _pos = lo + (hi - lo) * frac
                sc2d = _frame_vals.get(round(frac, 6))
                if sc2d is None or not np.isfinite(sc2d).any():
                    return [], _pos, frac*100.0
                _pc = np.full((_N, _N), np.float32(_pos))
                C = {n_ax: _pc, in_ax[0]: _G1f, in_ax[1]: _G2f}
                _traces = []
                _first_t = True
                for (_ox, _oy) in _offsets:
                    _mk = dict(
                        x=C[0] + np.float32(_ox), y=C[1] + np.float32(_oy),
                        z=C[2], surfacecolor=sc2d,
                        colorscale=cmap, cmin=_cmin, cmax=_cmax,
                        lighting=dict(ambient=0.95, diffuse=0.15, specular=0.0),
                        showlegend=False, showscale=_first_t,
                        hovertemplate=f"{field}: %{{surfacecolor:.4f}} {unit}"
                                      "<extra></extra>",
                    )
                    if _first_t:
                        _mk["colorbar"] = dict(
                            title=dict(text=f"{field} [{unit}]", side="right",
                                       font=dict(size=12, color="black")),
                            thickness=14, len=0.75,
                            tickfont=dict(size=10, color="black"),
                            outlinecolor="#333", outlinewidth=1)
                    _traces.append(go.Surface(**_mk))
                    _first_t = False
                return _traces, _pos, frac*100.0

            # ── 초기 표시용 슬라이스 (프레임 목록의 중앙 위치) ────────────────
            _init_frac = float(_fracs[len(_fracs)//2]) if n_frames > 1 \
                else float(_fracs[0])
            _init_slice_traces, pos, pct = _make_slice_traces(_init_frac)
            if not _init_slice_traces:
                return None
            # 유선(streamlines) 씨앗 위치용
            origin = {"x": (pos, cy, cz), "y": (cx, pos, cz),
                      "z": (cx, cy, pos)}.get(slice_normal, (cx, pos, cz))

            fig = go.Figure()
            # 슬라이스 트레이스 (프레임에서 교체될 트레이스)
            _slice_trace_indices = []
            for t in _init_slice_traces:
                _slice_trace_indices.append(len(fig.data))
                fig.add_trace(t)
            # STL 형상 (고정 — 프레임과 무관). 메시 경계 패치가 추출되면 그것을,
            # 아니면 STL 파일을 직접 읽어 표시한다(항목4: 모든 모드에서 STL 렌더).
            if _patches:
                for (ox, oy) in _offsets:
                    for (pp, pf) in _patches:
                        fig.add_trace(go.Mesh3d(
                            x=pp[:,0]+ox, y=pp[:,1]+oy, z=pp[:,2],
                            i=pf[:,0], j=pf[:,1], k=pf[:,2],
                            color='#888', opacity=float(stl_opacity),
                            showscale=False, showlegend=False, hoverinfo='skip',
                        ))
            else:
                for t in self._stl_traces(go, _offsets, opacity=stl_opacity):
                    fig.add_trace(t)

            # 유선 (Scatter3d 라인) — v13 항목11: 종전엔 n_frames<=1 조건 때문에
            # 슬라이스(n_frames=33)에서 유선 체크박스를 켜도 절대 그려지지 않았다.
            # 조건을 제거하고, seed 를 물체(그물) 상류의 격자 평면에 배치해 그물을
            # 통과하는 유선을 안정적으로 그린다(고정 트레이스 — 슬라이더와 무관).
            if show_streamlines and PYVISTA_OK and "U" in internal.array_names:
                try:
                    _sf = self._streamline_trace(pv, go, internal, bounds)
                    if _sf is not None:
                        fig.add_trace(_sf)
                except Exception as _e:
                    logger.warning(f"유선 렌더 실패: {_e}")

            # v10 항목4: 유동방향 화살표(고정 트레이스 — 프레임 교체와 무관)
            for t in self._flow_arrow_traces(go, bounds):
                fig.add_trace(t)

            # ── 다중 프레임 + Plotly 슬라이더 ───────────────────────────────
            _plotly_sliders = []
            if n_frames > 1:
                # v10 항목2·3: 적응 분율(위에서 배치 샘플링에 사용한 것 재사용)
                _frames = []
                for _frac in _fracs:
                    _ftr, _fpos, _fpct = _make_slice_traces(_frac)
                    if not _ftr:
                        continue
                    _ann_txt = (f"Slice {slice_normal.upper()} = {_fpos:.4f} m  ({_fpct:.1f}%)"
                                + (f"   |  타일 {tile_nx}×{tile_ny}"
                                   if tile_nx * tile_ny > 1 else ""))
                    _frames.append(go.Frame(
                        data=_ftr,
                        traces=_slice_trace_indices,
                        layout=go.Layout(annotations=[dict(
                            text=_ann_txt, xref="paper", yref="paper",
                            x=0.01, y=0.99, xanchor="left", yanchor="top",
                            showarrow=False, font=dict(size=12, color="black"),
                            bgcolor="rgba(255,255,255,0.92)", borderpad=4,
                            bordercolor="#333", borderwidth=1)]),
                        name=f"{_fpct:.1f}",
                    ))
                fig.frames = _frames
                _active_idx = len(_frames) // 2
                _plotly_sliders = [dict(
                    active=_active_idx,
                    pad=dict(b=10, t=10), len=0.9, x=0.05, y=0,
                    # 항목3: 슬라이더 대비 강화 — 진한 글자·그립·테두리·눈금.
                    bgcolor="#1a4a8a", bordercolor="#10243e", borderwidth=1,
                    tickcolor="#10243e", tickwidth=1, font=dict(color="black", size=11),
                    currentvalue=dict(
                        prefix=f"Slice {slice_normal.upper()} ",
                        suffix="%", visible=True, xanchor="right",
                        font=dict(size=12, color="black")),
                        transition=dict(duration=0),
                    # v10 항목3: 프레임 수 3배로 라벨이 겹치므로 4개마다 1개만 표기
                    # (currentvalue 에 정확한 % 상시 표시).
                    steps=[dict
                    (
                        method="animate",
                        args=[
                            [f.name],
                            dict
                            (
                                mode="immediate",
                                frame=dict(
                                    duration=0,
                                    redraw=True
                                ),
                                transition=dict(duration=0)
                            )

                        ],
                        label=(f.name if (_fi % max(1, len(fig.frames)//8) == 0)
                               else ""),
                    ) for _fi, f in enumerate(fig.frames)],
                )]

            # v11 항목3: 씬 범위를 관심영역(형상+후류) 중심으로 — 해석 대상이
            # 열자마자 화면 중앙에 크게 보이도록 한다(수동 이동 불필요).
            _vc, _vh = self._view_box(bounds)
            tiled_cx = _vc[0] + (tile_nx - 1) * dx / 2.0
            tiled_cy = _vc[1] + (tile_ny - 1) * dy / 2.0
            cz = _vc[2]
            _half = max(_vh,
                        (tile_nx * dx / 2.0) if tile_nx > 1 else 0.0,
                        (tile_ny * dy / 2.0) if tile_ny > 1 else 0.0, 1e-6)

            # 항목1: 슬라이스 모드는 배경 패널(박스 3면)을 끄고 단일 슬라이스만 표시.
            # 항목2: 축 제목·눈금 글자를 검정으로, 단위[m] 포함.
            _axttl = dict(color="black", size=12)
            _axtck = dict(color="black", size=10)
            scene = dict(
                xaxis=dict(title=dict(text="X [m]", font=_axttl),
                           range=[tiled_cx-_half, tiled_cx+_half],
                           visible=True, showticklabels=True, tickfont=_axtck,
                           backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="#b9c6d6", showbackground=False),
                yaxis=dict(title=dict(text="Y [m]", font=_axttl),
                           range=[tiled_cy-_half, tiled_cy+_half],
                           visible=True, showticklabels=True, tickfont=_axtck,
                           backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="#b9c6d6", showbackground=False),
                zaxis=dict(title=dict(text="Z [m]", font=_axttl),
                           range=[cz-_half, cz+_half],
                           visible=True, showticklabels=True, tickfont=_axtck,
                           backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="#b9c6d6", showbackground=False),
                aspectmode='cube',
                bgcolor='rgba(255,255,255,1)',
                uirevision='flowfield',
            )
            if init_camera:
                scene['camera'] = dict(eye=dict(x=1.0, y=1.0, z=1.0))
            _ann_init = (f"Slice {slice_normal.upper()} = {pos:.4f} m  ({pct:.1f}%)"
                         + (f"   |  타일 {tile_nx}×{tile_ny}"
                            if tile_nx * tile_ny > 1 else ""))
            _bottom = 60 if _plotly_sliders else 0

            _scene = dict(scene)


            fig.update_layout(
                uirevision='flowfield',
                annotations=[dict(
                    text=_ann_init, xref="paper", yref="paper",
                    x=0.01, y=0.99, xanchor="left", yanchor="top",
                    showarrow=False, font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.92)", borderpad=4,
                    bordercolor="#333", borderwidth=1)],
                #scene=scene,
                scene={
                    k:v
                    for k,v
                    in scene.items()
                    if k!="camera"
                },
                sliders=_plotly_sliders,
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=_bottom),
                height=520 + _bottom,
                paper_bgcolor='#f0f8ff',
            )
            return fig

        except Exception as e:
            logger.error(f"render_field_plotly 오류: {e}")
            return None

    # ─── 입체(비슬라이스) 등치면 + 애니메이션 (항목 13) ───────────────────

    def _internal_points_scalar(self, field: str):
        """내부 메시의 점 좌표와 스칼라(벡터는 크기)를 반환. (등치면용)"""
        mesh = self._get_mesh()
        if mesh is None:
            return None
        internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh
        ds = internal
        if field not in ds.array_names:
            return None
        try:
            if field in ds.point_data:
                pts, arr = ds.points, ds.point_data[field]
            else:
                ds2 = ds.cell_data_to_point_data()
                pts, arr = ds2.points, ds2[field]
        except Exception:
            pts, arr = ds.points, ds[field]
        vals = (np.linalg.norm(arr, axis=1) if getattr(arr, "ndim", 1) == 2
                else np.asarray(arr, float))
        return internal, np.asarray(pts), np.asarray(vals, float)

    def _stl_traces(self, go, offsets, opacity=0.15):
        """case 의 STL 형상(constant/triSurface/*.stl)을 직접 읽어 Mesh3d 로 반환.
        메시 경계 패치가 MultiBlock 이라 추출이 어려운 경우에도 STL 을 확실히
        표시한다(항목4). STL 은 보통 mm 단위이므로 메시(미터) bounds 와 비교해
        스케일을 자동 보정하고, 중심을 메시 중심에 맞춘다."""
        out = []
        if opacity is None or float(opacity) <= 0:
            return out
        try:
            import pyvista as pv
        except Exception:
            return out
        tri_dir = self.case_dir / "constant" / "triSurface"
        if not tri_dir.exists():
            return out
        mesh = self._get_mesh()
        internal = (mesh["internalMesh"] if mesh is not None
                    and "internalMesh" in mesh.keys() else None)
        mb = internal.bounds if internal is not None else None
        boundary = (mesh["boundary"] if mesh is not None
                    and "boundary" in mesh.keys() else None)

        def _patch_bounds(stem):
            """STL 파일명(stem)과 같은 이름의 경계 패치 bounds(표시 프레임).
            snappyHexMesh 가 STL 이름으로 벽 패치를 만들므로(netSurface.stl →
            netSurface) 이 패치가 형상의 실제 위치·크기의 기준이 된다."""
            if boundary is None:
                return None
            try:
                for k in boundary.keys():
                    if k.lower() == stem.lower():
                        p = boundary[k]
                        if p is not None and p.n_points > 0:
                            return p.bounds
            except Exception:
                pass
            return None

        for stl in sorted(tri_dir.glob("*.stl")):
            try:
                m = pv.read(str(stl))
                if m.n_points == 0:
                    continue
                mt = m.triangulate()
                pp = np.asarray(mt.points, float)
                pf = mt.faces.reshape(-1, 4)[:, 1:]
                sb = m.bounds
                sdiag = math.sqrt((sb[1]-sb[0])**2 + (sb[3]-sb[2])**2
                                  + (sb[5]-sb[4])**2)
                s_ctr = np.array([(sb[0]+sb[1])/2, (sb[2]+sb[3])/2,
                                  (sb[4]+sb[5])/2])
                pb = _patch_bounds(stl.stem)
                if pb is not None:
                    # 1순위: 같은 이름의 메시 경계 패치에 정합 — 도메인 크기와
                    # 무관하게 실제 형상 위치·스케일과 일치(전체구조의 대형·비대칭
                    # 도메인에서도 정확). 패치는 이미 표시 프레임이므로 STL 을
                    # 자기 중심 기준 R 회전 후 패치 중심에 배치한다.
                    pdiag = math.sqrt((pb[1]-pb[0])**2 + (pb[3]-pb[2])**2
                                      + (pb[5]-pb[4])**2)
                    sc = (pdiag / sdiag) if sdiag > 1e-12 else 1.0
                    p_ctr = np.array([(pb[0]+pb[1])/2, (pb[2]+pb[3])/2,
                                      (pb[4]+pb[5])/2])
                    pp = (pp - s_ctr) * sc
                    if getattr(self, "_disp_M", None) is not None:
                        _R = np.asarray(self._disp_M[:3, :3], float)
                        pp = pp @ _R.T
                    pp = pp + p_ctr
                elif mb is not None:
                    # 폴백(패치 미검출): 종전 휴리스틱 — 단위 스케일 자동 보정
                    # (예: mm→m) + 도메인 중심 정렬. 스케일은 바운딩박스
                    # '대각선'(회전 불변)으로 산정.
                    mdiag = math.sqrt((mb[1]-mb[0])**2 + (mb[3]-mb[2])**2
                                      + (mb[5]-mb[4])**2)
                    sc = (mdiag / sdiag) if sdiag > 1e-12 else 1.0
                    m_ctr = np.array([(mb[0]+mb[1])/2, (mb[2]+mb[3])/2,
                                      (mb[4]+mb[5])/2])
                    pp = (pp - s_ctr) * sc + m_ctr
                    # 통일 표시 프레임 회전(메시와 동일: 중심 기준 R 적용)
                    if getattr(self, "_disp_M", None) is not None:
                        _R = np.asarray(self._disp_M[:3, :3], float)
                        pp = (pp - m_ctr) @ _R.T + m_ctr
                for (ox, oy) in offsets:
                    out.append(go.Mesh3d(
                        x=pp[:, 0]+ox, y=pp[:, 1]+oy, z=pp[:, 2],
                        i=pf[:, 0], j=pf[:, 1], k=pf[:, 2],
                        color='#777', opacity=float(opacity),
                        showscale=False, showlegend=False, hoverinfo='skip'))
            except Exception:
                pass
        return out

    def _boundary_traces(self, go, offsets, opacity=0.15):
        """STL 형상(메시 경계면) trace 목록 — 타일 오프셋 포함. opacity 로 가시성 조절."""
        mesh = self._get_mesh()
        out = []
        if mesh is None:
            return out
        for key in mesh.keys():
            if key == "internalMesh":
                continue
            try:
                patch = mesh[key]
                if patch.n_points == 0:
                    continue
                ptri = patch.triangulate()
                pf = ptri.faces.reshape(-1, 4)[:, 1:]
                pp = ptri.points
                for (ox, oy) in offsets:
                    out.append(go.Mesh3d(
                        x=pp[:, 0]+ox, y=pp[:, 1]+oy, z=pp[:, 2],
                        i=pf[:, 0], j=pf[:, 1], k=pf[:, 2],
                        color='#888', opacity=float(opacity),
                        showscale=False, showlegend=False, hoverinfo='skip'))
            except Exception:
                pass
        # 메시 경계 패치 추출이 비면(예: MultiBlock 'boundary') STL 파일로 폴백(항목4)
        if not out:
            return self._stl_traces(go, offsets, opacity=opacity)
        return out

    def _view_box(self, bounds):
        """v11 항목3: 초기 뷰 박스 — 관심영역(형상+후류) 중심과 반경.

        내부 장애물 패치 기반 _focus_box 가 있으면 그 중심·크기(×1.15 여유)를,
        없으면 도메인 전체를 사용한다. 전체구조처럼 도메인이 형상보다 훨씬 클 때
        해석 대상이 화면 중앙에 크게 보이도록 한다."""
        ex = [max(bounds[1]-bounds[0], 1e-9), max(bounds[3]-bounds[2], 1e-9),
              max(bounds[5]-bounds[4], 1e-9)]
        c = np.array([(bounds[0]+bounds[1])/2.0, (bounds[2]+bounds[3])/2.0,
                      (bounds[4]+bounds[5])/2.0])
        half = max(ex) / 2.0
        # 내부 장애물(그물) 패치 bbox 를 '물체 중심'으로 사용, 뷰 크기는 물체의
        # ~3.2배(후류 일부 포함, 물체가 화면의 ~1/3 차지). 커서 중심 줌으로
        # 원방 확인은 쉬우므로 초기값은 물체 가시성을 우선한다.
        try:
            mesh = self._mesh_cache
            boundary = (mesh["boundary"] if mesh is not None
                        and "boundary" in mesh.keys() else None)
            if boundary is not None:
                diag = math.sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)
                tol = 1e-3 * diag
                lo = np.array([bounds[0], bounds[2], bounds[4]], float)
                hi = np.array([bounds[1], bounds[3], bounds[5]], float)
                blo = bhi = None
                for k in boundary.keys():
                    try:
                        p = boundary[k]
                        if p is None or p.n_points == 0:
                            continue
                        pb = p.bounds
                    except Exception:
                        continue
                    plo = np.array([pb[0], pb[2], pb[4]], float)
                    phi = np.array([pb[1], pb[3], pb[5]], float)
                    if (plo > lo + tol).all() and (phi < hi - tol).all():
                        blo = plo if blo is None else np.minimum(blo, plo)
                        bhi = phi if bhi is None else np.maximum(bhi, phi)
                if blo is not None:
                    L = float(np.max(bhi - blo))
                    if L > 1e-9:
                        c = (blo + bhi) / 2.0
                        half = min(1.6 * L, max(ex) / 2.0)
        except Exception:
            pass
        return c, max(half, 1e-6)

    def _iso_scene(self, go, bounds, tile_nx, tile_ny, dx, dy,
                   init_camera=True, anim=None):
        """등치면·입체 뷰의 scene(축·카메라) 레이아웃.

        uirevision='flowfield' 고정으로 슬라이스·등치면 모드 전환 시 카메라 보존.
        등치면 스윕 재생 중 카메라 복원은 JS iframe 핸들러(plotly_buttonclicked)가 담당.
        """
        # v11 항목3: 관심영역(형상+후류) 중심 초기 뷰 — 대상이 화면 중앙에 크게.
        _vc, _vh = self._view_box(bounds)
        cx = _vc[0] + (tile_nx-1)*dx/2.0
        cy = _vc[1] + (tile_ny-1)*dy/2.0
        cz = _vc[2]
        _half = max(_vh,
                    (tile_nx*dx/2.0) if tile_nx > 1 else 0.0,
                    (tile_ny*dy/2.0) if tile_ny > 1 else 0.0, 1e-6)
        # 항목1/일관성: 배경 패널(박스 면) 제거. 항목2: 축 글자 검정·단위[m].
        _axttl = dict(color="black", size=12)
        _axtck = dict(color="black", size=10)
        scene = dict(
            xaxis=dict(title=dict(text="X [m]", font=_axttl),
                       range=[cx-_half, cx+_half], visible=True,
                       showticklabels=True, tickfont=_axtck,
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#b9c6d6", showbackground=False),
            yaxis=dict(title=dict(text="Y [m]", font=_axttl),
                       range=[cy-_half, cy+_half], visible=True,
                       showticklabels=True, tickfont=_axtck,
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#b9c6d6", showbackground=False),
            zaxis=dict(title=dict(text="Z [m]", font=_axttl),
                       range=[cz-_half, cz+_half], visible=True,
                       showticklabels=True, tickfont=_axtck,
                       backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#b9c6d6", showbackground=False),
            aspectmode='cube', bgcolor='rgba(255,255,255,1)',
        )
        # 슬라이스·등치면 모두 동일한 uirevision으로 모드 전환 시 카메라 보존.
        # 등치면 스윕 재생 중 카메라 복원은 JS 핸들러(plotly_buttonclicked)가 담당.
        scene['uirevision'] = 'flowfield'
        
        # if anim != 'sweep' and init_camera:
        #     scene['camera'] = dict(eye=dict(x=1.0, y=1.0, z=1.0))
            # 최초 생성 시에만 기본 카메라 적용
        if init_camera and 'camera' not in scene:
            scene.setdefault(
                'camera',
                dict(
                    eye=dict(
                        x=1.0,
                        y=1.0,
                        z=1.0
                    )
                )
            )
        return scene

    def _focus_box(self, mesh, b, ex):
        """형상(내부 장애물 패치) 기반 관심영역 박스 (flo, fhi) 반환.

        도메인 경계에 닿지 않는 벽 패치(netSurface 등)의 합집합 bbox 를,
        표시 프레임 유동방향 d(AoA) 기준으로 상류 1L·하류(후류) 3L·측방 0.5L
        확장한 뒤 도메인으로 클립한다. 형상 집중 리샘플 격자(_focus_grid)·
        적응형 슬라이스 분율·연속 볼륨 모드가 공용한다.
        내부 패치 미검출/형상이 도메인 대부분이면 None."""
        boundary = (mesh["boundary"] if mesh is not None
                    and "boundary" in mesh.keys() else None)
        if boundary is None:
            return None
        diag = math.sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)
        tol = 1e-3 * diag
        lo = np.array([b[0], b[2], b[4]], float)
        hi = np.array([b[1], b[3], b[5]], float)
        fb_lo = fb_hi = None
        for k in boundary.keys():
            try:
                p = boundary[k]
                if p is None or p.n_points == 0:
                    continue
                pb = p.bounds
            except Exception:
                continue
            plo = np.array([pb[0], pb[2], pb[4]], float)
            phi = np.array([pb[1], pb[3], pb[5]], float)
            # 도메인 경계에 닿지 않는 패치 = 내부 장애물(그물/가두리)
            if (plo > lo + tol).all() and (phi < hi - tol).all():
                fb_lo = plo if fb_lo is None else np.minimum(fb_lo, plo)
                fb_hi = phi if fb_hi is None else np.maximum(fb_hi, phi)
        if fb_lo is None:
            return None
        L = float(np.linalg.norm(fb_hi - fb_lo))
        if L < 1e-9 or L > 0.7 * diag:   # 형상이 도메인 대부분이면 무의미
            return None
        # 표시 프레임 유동방향 d(AoA): 하류(후류) 3L·상류 1L·측방 0.5L 연장
        aa = math.radians(self.angle_deg if self.angle_deg is not None else 90.0)
        d = np.array([math.sin(aa), -math.cos(aa), 0.0])
        dpos, dneg = np.maximum(0.0, d), np.maximum(0.0, -d)
        flo = np.maximum(fb_lo - L*(0.5 + 1.0*dpos + 3.0*dneg), lo)
        fhi = np.minimum(fb_hi + L*(0.5 + 3.0*dpos + 1.0*dneg), hi)
        return flo, fhi

    def _focus_grid(self, mesh, b, ex):
        """입체/등치면 리샘플용 '형상 집중' 비균일 rectilinear 격자.

        고정 예산의 균일 격자는 전체구조처럼 도메인(≈1m)이 형상(≈0.1m)보다 훨씬
        클 때 복셀이 ~25mm 로 굵어져, 등치면이 물리와 무관한 각진 덩어리(마칭큐브
        앨리어싱)로 나온다. 관심영역(_focus_box)에 격자를 집중(~10mm 이하)하고
        원방(자유류, |U| 균일)은 성기게 둔다.
        관심영역 미검출이면 None(종전 균일 격자 폴백)."""
        import pyvista as pv
        fb = self._focus_box(mesh, b, ex)
        if fb is None:
            return None
        flo, fhi = fb
        diag = math.sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)
        tol = 1e-3 * diag
        lo = np.array([b[0], b[2], b[4]], float)
        hi = np.array([b[1], b[3], b[5]], float)
        fex = np.maximum(fhi - flo, 1e-9)
        h = (float(np.prod(fex)) / 100000.0) ** (1.0/3.0)   # 세밀부 ~10만 점
        axes = []
        for i in range(3):
            h_ax = max(h, float(fex[i]) / 100.0)   # 축당 세밀 노드 ≤ ~100
            n = max(8, int(round(float(fex[i]) / h_ax)) + 1)
            fine = np.linspace(flo[i], fhi[i], n)
            pre = (np.linspace(lo[i], flo[i], 7, endpoint=False)
                   if flo[i] - lo[i] > tol else np.empty(0))
            post = (np.linspace(fhi[i], hi[i], 8)[1:]
                    if hi[i] - fhi[i] > tol else np.empty(0))
            axes.append(np.concatenate([pre, fine, post]))
        return pv.RectilinearGrid(axes[0], axes[1], axes[2])

    @staticmethod
    def _alpha_colorscale(cmap_name: str, alpha: float):
        """컬러스케일 각 색에 알파를 입힌 rgba 스케일 (v12 항목4).

        투명도 슬라이더 값을 렌더링과 컬러바가 '같은 소스'로 공유하게 한다 —
        트레이스에 이 스케일을 쓰면 표면 투명도와 우측 컬러바 색이 항상 일치."""
        import plotly.colors as pc
        a = min(max(float(alpha), 0.0), 1.0)
        try:
            base = pc.get_colorscale(cmap_name)
        except Exception:
            base = pc.get_colorscale("Jet")
        out = []
        for pos, col in base:
            col = str(col)
            try:
                if col.startswith("#"):
                    r, g, b = pc.hex_to_rgb(col)
                else:   # 'rgb(r,g,b)' / 'rgba(r,g,b,a)'
                    nums = col[col.find("(")+1:col.find(")")].split(",")
                    r, g, b = (int(float(v)) for v in nums[:3])
            except Exception:
                r, g, b = 128, 128, 128
            out.append([float(pos), f"rgba({r},{g},{b},{a:.3f})"])
        return out

    def _streamline_trace(self, pv, go, internal, bounds):
        """v13 항목11: 물체(그물) 상류 격자 seed → 통과 유선 Scatter3d.

        표시 프레임 유동방향 d(AoA) 기준으로 물체 상류 0.5L 평면에 격자 seed 를
        깔아 그물을 통과·우회하는 유선을 그린다. seed 를 물체 크기에 맞추므로
        도메인이 큰 전체구조에서도 유선이 형상 주변에 모인다."""
        _vc, _vh = self._view_box(bounds)
        ang = self.angle_deg if self.angle_deg is not None else 90.0
        aa = math.radians(float(ang))
        d = np.array([math.sin(aa), -math.cos(aa), 0.0])  # 표시=솔버 근사(축 정렬 도메인)
        # 솔버 좌표 유동방향(seed 평면 법선) — 실제 U 방향으로 상류를 잡는다
        Uc = np.asarray(internal.cell_data["U"] if "U" in internal.cell_data
                        else internal.point_data["U"], float)
        Um = Uc.mean(axis=0)
        n = Um / (np.linalg.norm(Um) + 1e-12)
        # 상류 평면 중심: 물체 중심에서 유동 반대로 0.6L
        ctr = np.array(_vc) - n * (1.2 * _vh)
        # 평면 내 두 직교축
        _t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(n, _t); e1 /= (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(n, e1)
        g = np.linspace(-0.9 * _vh, 0.9 * _vh, 9)
        seeds_pts = np.array([ctr + a * e1 + b * e2 for a in g for b in g])
        seeds = pv.PolyData(seeds_pts)
        stream = internal.streamlines_from_source(
            seeds, vectors="U",
            integration_direction="both",
            max_time=None, max_steps=2000,
            initial_step_length=0.05, terminal_speed=1e-6)
        if stream is None or stream.n_points == 0:
            return None
        # 라인 세그먼트를 NaN 구분으로 이어붙여 단일 Scatter3d 로 (성능)
        try:
            lines = stream.lines
            pts = np.asarray(stream.points)
            xs, ys, zs = [], [], []
            i = 0
            while i < len(lines):
                npt = lines[i]
                idx = lines[i+1:i+1+npt]
                seg = pts[idx]
                xs.extend(seg[:, 0].tolist() + [np.nan])
                ys.extend(seg[:, 1].tolist() + [np.nan])
                zs.extend(seg[:, 2].tolist() + [np.nan])
                i += npt + 1
        except Exception:
            pts = np.asarray(stream.points)
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#111", width=2),
            name="유선", showlegend=False, hoverinfo="skip")

    def _flow_arrow_traces(self, go, bounds):
        """유동방향 표시기(v10 항목4): 표시 프레임 d(AoA) 3D 화살표 + 라벨.

        도메인 상부(z_max 위)에 도메인 중심을 향하는 진홍색 화살표를 그려
        카메라 방향과 무관하게 유동 방향을 즉시 식별하게 한다. 3모드 공통."""
        out = []
        try:
            ang = self.angle_deg
            if ang is None or self.mode not in ("unit_cell", "full_structure"):
                return out
            aa = math.radians(float(ang))
            d = np.array([math.sin(aa), -math.cos(aa), 0.0])
            # v11 항목3: 뷰 박스(관심영역) 기준 배치 — 초기 뷰가 물체 중심으로
            # 좁아져도 화살표가 항상 시야 안(물체 상공)에 위치.
            _vc, _vh = self._view_box(bounds)
            c = np.array([_vc[0], _vc[1], _vc[2] + 0.72*_vh])
            A = 0.55 * _vh
            p0, p1 = c - d*A, c            # 꼬리→머리(도메인 중심 상공)
            col = "#c8102e"
            out.append(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                mode="lines", line=dict(color=col, width=8),
                showlegend=False, hoverinfo="skip"))
            out.append(go.Cone(
                x=[p1[0]], y=[p1[1]], z=[p1[2]],
                u=[d[0]], v=[d[1]], w=[d[2]],
                sizemode="absolute", sizeref=0.30*A, anchor="tip",
                colorscale=[[0, col], [1, col]], showscale=False,
                hoverinfo="skip"))
            out.append(go.Scatter3d(
                x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="text",
                text=[f"유동 (AoA {float(ang):.0f}°)"],
                textfont=dict(size=13, color=col), textposition="top center",
                showlegend=False, hoverinfo="skip"))
        except Exception:
            return []
        return out

    def _adaptive_fracs(self, mesh, slice_normal, bounds, n_frames):
        """v10 항목2·3: 슬라이스 위치 분율 — 형상 주변 세밀 + 원방 성김.

        관심영역(_focus_box)이 슬라이스 축에서 차지하는 구간에 스텝의 ~70%를
        집중 배치해, 도메인이 형상보다 훨씬 큰 전체구조에서도 그물 주변을
        미세 간격으로 통과한다. 관심영역 미검출 시 균일 분포."""
        n = max(2, int(n_frames))
        ax = {"x": 0, "y": 1, "z": 2}.get(slice_normal, 1)
        lo, hi = bounds[2*ax], bounds[2*ax+1]
        span = max(hi - lo, 1e-12)
        fb = None
        try:
            ex = [max(bounds[1]-bounds[0], 1e-9), max(bounds[3]-bounds[2], 1e-9),
                  max(bounds[5]-bounds[4], 1e-9)]
            fb = self._focus_box(mesh, bounds, ex)
        except Exception:
            fb = None
        if fb is None:
            fr = np.linspace(0.05, 0.95, n)
        else:
            a = max(0.02, (float(fb[0][ax]) - lo) / span)
            b2 = min(0.98, (float(fb[1][ax]) - lo) / span)
            if not (b2 > a):
                fr = np.linspace(0.05, 0.95, n)
            else:
                n_fine = max(2, int(round(n * 0.7)))
                n_coarse = max(2, n - n_fine)
                fr = np.concatenate([np.linspace(0.02, 0.98, n_coarse),
                                     np.linspace(a, b2, n_fine)])
        # v12 항목3: 물체 기하중심을 지나는 슬라이스를 각 축에서 항상 포함
        try:
            _vc, _ = self._view_box(bounds)
            _cf = (float(_vc[ax]) - lo) / span
            if 0.0 <= _cf <= 1.0:
                fr = np.append(fr, _cf)
        except Exception:
            pass
        return np.unique(np.round(fr, 4))

    def render_field_volume(self, field: str = "U",
                            opacity: float = 0.12,
                            surface_count: int = 17,
                            stl_opacity: float = 0.15,
                            init_camera: bool = True,
                            tile_nx: int = 1, tile_ny: int = 1,
                            clip: Optional[Any] = None,
                            clip_planes: bool = True
                            ) -> Optional[Any]:
        """v10 항목1: 슬라이스 사이를 보간한 **연속 볼륨** 뷰(go.Volume).

        관심영역(_focus_box: 그물 주변 상류1L·하류3L·측방0.5L, 미검출 시 도메인
        전체)을 균일 격자(~10만 점)로 리샘플해 반투명 연속 볼륨으로 표시한다.
        원방은 균일 자유류라 표시 생략해도 정보 손실이 없다. 리샘플은 케이스당
        1회 캐시(_vol_cache). 타일링은 데이터량 문제로 미지원(단일 표시)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None
        if not PYVISTA_OK:
            return None
        try:
            import pyvista as pv
            mesh = self._get_mesh()
            if mesh is None:
                return None
            internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh
            if field not in internal.array_names:
                return None
            b = internal.bounds
            ex = [max(b[1]-b[0], 1e-9), max(b[3]-b[2], 1e-9), max(b[5]-b[4], 1e-9)]
            _vc = getattr(self, "_vol_cache", None)
            if _vc is not None and _vc[0] == self._cache_time:
                sampled, lo3, hi3, dims = _vc[1:]
            else:
                fb = None
                try:
                    fb = self._focus_box(mesh, b, ex)
                except Exception:
                    fb = None
                if fb is not None:
                    lo3, hi3 = fb
                else:
                    lo3 = np.array([b[0], b[2], b[4]], float)
                    hi3 = np.array([b[1], b[3], b[5]], float)
                fex = np.maximum(hi3 - lo3, 1e-9)
                h = (float(np.prod(fex)) / 100000.0) ** (1.0/3.0)
                dims = tuple(int(np.clip(round(float(fex[i]) /
                                                max(h, float(fex[i])/96.0)) + 1,
                                          8, 96)) for i in range(3))
                g = pv.ImageData()
                g.dimensions = dims
                g.origin = tuple(lo3)
                g.spacing = tuple(float(fex[i])/max(1, dims[i]-1) for i in range(3))
                sampled = g.sample(internal)
                self._vol_cache = (self._cache_time, sampled, lo3, hi3, dims)
            arr = sampled[field]
            vals = (np.linalg.norm(arr, axis=1) if getattr(arr, "ndim", 1) == 2
                    else np.asarray(arr, float))
            if "vtkValidPointMask" in sampled.array_names:
                _m = np.asarray(sampled["vtkValidPointMask"])
                vals = np.where(_m > 0, vals, np.nan)
            _valid = vals[~np.isnan(vals)]
            if _valid.size == 0:
                return None
            vmin, vmax = float(_valid.min()), float(_valid.max())
            vals = np.nan_to_num(vals, nan=vmin)
            # v11 항목5: 표시 범위 0 ~ 최대(저속 영역 포함). 음수 가능한 필드(p)는
            # 실제 최소 사용.
            lo_disp = 0.0 if vmin >= 0.0 else vmin
            # go.Volume 은 z-fastest 격자 순서를 가정 → pyvista(x-fastest)를 변환
            ax_lin = [np.linspace(lo3[i], hi3[i], dims[i]) for i in range(3)]
            X, Y, Z = np.meshgrid(*ax_lin, indexing="ij")
            value = vals.reshape(dims, order="F").ravel(order="C")
            # ── v13 항목15: XYZ 축별 '대칭' 클리핑 — clip=((lo,hi)×3), 각 값은
            # [0,1] 정규화 위치(0=축 최소, 1=축 최대). 기본 (0,1)=전체 표시.
            # UI 의 -50~+50% 양방향 슬라이더가 이 (lo,hi) 로 매핑되어 음/양 양쪽에서
            # 독립·동시 절단할 수 있다. 임계 밖 값을 isomin 아래로 밀어 렌더 제외
            # (컬러맵·불투명도 전달함수 보존, 내부 단면 자동 노출).
            _clip_bounds = None
            if clip is not None:
                _norm = []
                for _c in clip:
                    if isinstance(_c, (tuple, list)):
                        _lo, _hi = float(_c[0]), float(_c[1])
                    else:   # 하위호환: 단일값 = 상한만
                        _lo, _hi = 0.0, float(_c)
                    _norm.append((min(max(_lo, 0.0), 1.0),
                                  min(max(_hi, 0.0), 1.0)))
                _clip_bounds = _norm
                _hide = lo_disp - max(abs(vmax - lo_disp), 1e-6) * 0.05 - 1e-9
                _keep = np.ones(value.shape, dtype=bool)
                for _ci, (_arr3, (_lo, _hi)) in enumerate(zip((X, Y, Z), _norm)):
                    if _lo > 1e-6 or _hi < 0.9999:
                        _t0 = lo3[_ci] + _lo * (hi3[_ci] - lo3[_ci])
                        _t1 = lo3[_ci] + _hi * (hi3[_ci] - lo3[_ci])
                        _r = _arr3.ravel()
                        _keep &= (_r >= _t0 - 1e-12) & (_r <= _t1 + 1e-12)
                value = np.where(_keep, value, _hide)
            cmap = self._PLOTLY_CMAP.get(field, "Jet")
            unit = self._FIELD_UNIT.get(field, "")
            _a = min(max(float(opacity), 0.0), 1.0)
            # v13 항목14: 볼륨 opacity 의 의미를 명확히 한다. go.Volume 은 광선이
            # 통과하는 여러 반투명 층을 누적 블렌딩하므로, 단일 opacity 만으로는
            # 100% 여도 최대색이 범례(불투명)보다 연하게 보였다(사용자 지적).
            # opacityscale 로 '값이 높을수록 불투명' 전달함수를 주고, 슬라이더
            # 값(_a)을 최고값의 알파로 직접 매핑 → opacity 를 올리면 실제로
            # 불투명해지고(채도만이 아니라), 100% 에서 최대색이 범례색에 도달한다.
            _opsc = [[0.0, 0.0], [0.35, _a*0.35], [0.7, _a*0.75], [1.0, _a]]
            fig = go.Figure()
            fig.add_trace(go.Volume(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=value,
                isomin=lo_disp, isomax=vmax,
                cmin=lo_disp, cmax=vmax,
                opacity=1.0, opacityscale=_opsc,
                surface_count=max(int(surface_count), 21),
                colorscale=cmap,
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,      # 컬러바는 아래 동기 전용 트레이스가 담당
                hoverinfo="skip"))
            # v12 항목4 / v13 항목14: 투명도 동기 컬러바 — opacityscale 의 최고
            # 알파(_a)를 컬러바에도 그대로 입혀(rgba) 화면 표시색과 범례를 일치.
            fig.add_trace(go.Scatter3d(
                x=[lo3[0], lo3[0]], y=[lo3[1], lo3[1]], z=[lo3[2], lo3[2]],
                mode="markers",
                marker=dict(size=0.001, opacity=0.0,
                            color=[lo_disp, vmax],
                            colorscale=self._alpha_colorscale(cmap, _a),
                            cmin=lo_disp, cmax=vmax, showscale=True,
                            colorbar=dict(
                                title=dict(text=f"{field} [{unit}]",
                                           side="right",
                                           font=dict(size=12, color="black")),
                                thickness=14, len=0.75,
                                tickfont=dict(size=10, color="black"),
                                outlinecolor="#333", outlinewidth=1)),
                showlegend=False, hoverinfo="skip"))
            # v13 항목12: 클리핑 경계 미리보기 평면 — 활성(비-전체) 클리핑 축마다
            # 절단면 위치에 반투명 평면을 그려, 최종 렌더 전에도 어느 부분이
            # 잘리는지 즉시 보이게 한다(볼륨 리샘플은 캐시라 클리핑은 이미 실시간).
            if clip_planes and _clip_bounds is not None:
                _pcol = "#1a4a8a"
                for _ci, (_lo, _hi) in enumerate(_clip_bounds):
                    for _fr, _act in ((_lo, _lo > 1e-6), (_hi, _hi < 0.9999)):
                        if not _act:
                            continue
                        _p = lo3[_ci] + _fr * (hi3[_ci] - lo3[_ci])
                        _u = [(lo3[j], hi3[j]) for j in range(3)]
                        _u[_ci] = (_p, _p)
                        _gx, _gy, _gz = (np.array([[_u[0][0], _u[0][1]]]),
                                         None, None)
                        # 평면 4모서리
                        _oth = [j for j in range(3) if j != _ci]
                        _c0 = np.array([lo3[_oth[0]], hi3[_oth[0]]])
                        _c1 = np.array([lo3[_oth[1]], hi3[_oth[1]]])
                        A, Bp = np.meshgrid(_c0, _c1, indexing="ij")
                        P = np.zeros((2, 2, 3))
                        P[..., _ci] = _p
                        P[..., _oth[0]] = A
                        P[..., _oth[1]] = Bp
                        fig.add_trace(go.Surface(
                            x=P[..., 0], y=P[..., 1], z=P[..., 2],
                            surfacecolor=np.zeros((2, 2)),
                            colorscale=[[0, _pcol], [1, _pcol]],
                            showscale=False, opacity=0.18,
                            hoverinfo="skip", showlegend=False))
            for t in self._boundary_traces(go, [(0.0, 0.0)], opacity=stl_opacity):
                fig.add_trace(t)
            for t in self._flow_arrow_traces(go, b):
                fig.add_trace(t)
            _scene = self._iso_scene(go, b, 1, 1, ex[0], ex[1],
                                     init_camera=init_camera)
            _scene.pop("camera", None)   # 카메라 보존(영속 루프)과 동일 패턴
            fig.update_layout(
                uirevision='flowfield',
                annotations=[dict(
                    text="연속 볼륨 (관심영역: 형상 주변+후류)",
                    xref="paper", yref="paper", x=0.01, y=0.99,
                    xanchor="left", yanchor="top", showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.92)", borderpad=4,
                    bordercolor="#333", borderwidth=1)],
                scene=_scene,
                showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                height=520, paper_bgcolor='#f0f8ff')
            return fig
        except Exception as e:
            logger.error(f"연속 볼륨 렌더 오류: {e}")
            return None

    def render_field_3d(self, field: str = "U", level: Optional[float] = None,
                        level_frac: Optional[float] = None,
                        anim: Optional[str] = None,
                        tile_nx: int = 1, tile_ny: int = 1,
                        n_frames: int = 24,
                        init_camera: bool = True,
                        opacity: float = 0.55,
                        stl_opacity: float = 0.15) -> Optional[Any]:
        """슬라이스가 아닌 **입체 등치면(Isosurface)** 3D 뷰.

        anim=None  : 정적. level 지정 시 그 |U| 등치면 1개, 없으면 다중 등치면.
        anim='rotate': 카메라가 궤도를 도는 재생 버튼(입체 회전 동영상).
        anim='sweep' : 등치값을 낮은→높은 |U|로 자동 스윕하는 재생 버튼.
        tile_nx/ny : 주기 단위셀 시각화 복제(해석은 1셀).
        """
        try:
            import plotly.graph_objects as go
            import math
        except ImportError:
            return None
        if not PYVISTA_OK:
            return None
        try:
            import pyvista as pv
            mesh = self._get_mesh()
            if mesh is None:
                return None
            internal = mesh["internalMesh"] if "internalMesh" in mesh.keys() else mesh
            if field not in internal.array_names:
                return None
            b = internal.bounds
            ex = [max(b[1]-b[0], 1e-9), max(b[3]-b[2], 1e-9), max(b[5]-b[4], 1e-9)]
            # 내부 메시 점 전체를 그대로 넘기면 figure가 수백 MB가 되어 웹소켓
            # 한도를 초과한다. 형상(그물) 주변 집중 비균일 격자(_focus_grid)로
            # 리샘플하고, 미적용 시 도메인 비율 균일 격자(~4.5만 점) 폴백.
            # 리샘플은 필드 무관(전 필드 포함)이라 메시 캐시와 같은 수명으로 캐시.
            sampled = None
            _rs = getattr(self, "_rs_cache", None)
            if _rs is not None and _rs[0] == self._cache_time:
                sampled = _rs[1]
            if sampled is None:
                grid = None
                try:
                    grid = self._focus_grid(mesh, b, ex)
                except Exception as _e:
                    logger.warning(f"형상 집중 격자 생성 실패(균일 폴백): {_e}")
                if grid is None:
                    _budget = 45000
                    _scale = (_budget / (ex[0]*ex[1]*ex[2])) ** (1.0/3.0)
                    _dims = [max(6, min(110, int(round(e*_scale)))) for e in ex]
                    grid = pv.ImageData()
                    grid.dimensions = _dims
                    grid.origin = (b[0], b[2], b[4])
                    grid.spacing = (ex[0]/max(1, _dims[0]-1),
                                    ex[1]/max(1, _dims[1]-1),
                                    ex[2]/max(1, _dims[2]-1))
                sampled = grid.sample(internal)
                self._rs_cache = (self._cache_time, sampled)
            if field not in sampled.array_names:
                return None
            arr = sampled[field]
            vals = (np.linalg.norm(arr, axis=1) if getattr(arr, "ndim", 1) == 2
                    else np.asarray(arr, float))
            if "vtkValidPointMask" in sampled.array_names:
                _m = np.asarray(sampled["vtkValidPointMask"])
                vals = np.where(_m > 0, vals, np.nan)
            pts = np.asarray(sampled.points)
            _valid = vals[~np.isnan(vals)]
            if _valid.size == 0:
                return None
            vmin, vmax = float(_valid.min()), float(_valid.max())
            if not (vmax > vmin):
                vmax = vmin + 1e-6
            # v11 항목4·5: 표시 범위는 항상 0 ~ 최대값 — 슬라이더·등치값·컬러맵이
            # 전체 속도 분포(0 m/s 포함)를 나타내야 한다. (v10 의 관심영역 분위수
            # 레벨은 FS 에서 0.899 부터 시작하는 문제를 유발해 폐기.)
            lo_disp = 0.0 if vmin >= 0.0 else vmin   # |U| 등은 0, 압력 등은 실제 최소
            cmin_disp = lo_disp

            dx, dy = ex[0], ex[1]
            tile_nx = max(1, int(tile_nx)); tile_ny = max(1, int(tile_ny))
            offsets = [(i*dx, j*dy) for i in range(tile_nx) for j in range(tile_ny)]
            if level is None and level_frac is not None:
                level = vmin + float(level_frac) * (vmax - vmin)
            cmap = self._PLOTLY_CMAP.get(field, "Jet")
            unit = self._FIELD_UNIT.get(field, "")
            # v12 항목4: 투명도 동기용 rgba 스케일(표면·컬러바 공통 소스)
            _cscale_a = self._alpha_colorscale(cmap, opacity)

            # 등치면 렌더링: go.Isosurface는 plotly가 격자를 특정 순서(z-fastest)로
            # 재구성한다고 가정하는데, pyvista ImageData의 점 순서는 x-fastest라
            # 표면 폴리곤이 만들어지지 않는다(컬러바만 보이고 메시 안 보임). 따라서
            # pyvista .contour()로 등치면을 직접 삼각화해, 슬라이스 모드와 동일한
            # 명시 면 인덱스(i,j,k) go.Mesh3d로 그린다(확실히 렌더링됨).
            sampled["__mag__"] = vals

            def _contour_geom(lv):
                """리샘플 그리드에서 |field|=lv 등치면을 (점, 삼각형면, 색강도)로 추출."""
                try:
                    cont = sampled.contour(isosurfaces=[float(lv)], scalars="__mag__")
                except Exception:
                    return None
                if cont is None or cont.n_points == 0:
                    return None
                ct = cont.triangulate()
                raw = ct.faces
                if len(raw) == 0:
                    return None
                # 페이로드 절감 1: 대형 표면 데시메이션 — 등치면 위 스칼라는
                # 상수(=lv)라 색 손실 없이 삼각형 수만 줄인다(레벨 48개 스윕 대응).
                if ct.n_cells > 12000:
                    try:
                        ct = ct.decimate_pro(1.0 - 12000.0 / ct.n_cells)
                        raw = ct.faces
                        if len(raw) == 0:
                            return None
                    except Exception:
                        pass
                # 페이로드 절감 2: float32/int32 다운캐스트(바이너리 직렬화 ~1/2).
                cp = np.asarray(ct.points, dtype=np.float32)
                cf = raw.reshape(-1, 4)[:, 1:].astype(np.int32)
                cintens = np.full(cp.shape[0], float(lv), dtype=np.float32)
                return cp, cf, cintens

            def _mesh(geom, ox, oy, first):
                """등치면 지오메트리를 타일 오프셋 적용해 go.Mesh3d로 변환.
                geom=None(빈 등치면)이면 빈 메시로 트레이스 자리만 유지(프레임 일관성)."""
                if geom is None:
                    return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[],
                                     showscale=False, showlegend=False,
                                     hoverinfo="skip")
                cp, cf, cintens = geom
                kw = dict(
                    x=cp[:, 0]+ox, y=cp[:, 1]+oy, z=cp[:, 2],
                    i=cf[:, 0], j=cf[:, 1], k=cf[:, 2],
                    intensity=cintens,
                    # v12 항목4: 투명도를 rgba 컬러스케일로 적용(opacity 속성 대신)
                    # → 표면 투명도와 우측 컬러바가 항상 동기화된다.
                    colorscale=_cscale_a,
                    cmin=cmin_disp, cmax=vmax,   # v11 항목5: 컬러맵 0~최대
                    opacity=1.0, flatshading=False, showscale=first,
                    showlegend=False,
                    hovertemplate=f"{field}: %{{intensity:.4f}} {unit}<extra></extra>")
                if first:
                    kw["colorbar"] = dict(
                        title=dict(text=f"{field} [{unit}]", side="right",
                                   font=dict(size=12, color="black")),
                        thickness=14, len=0.75,
                        tickfont=dict(size=10, color="black"),
                        outlinecolor="#333", outlinewidth=1)
                return go.Mesh3d(**kw)

            fig = go.Figure()
            n_tiles = len(offsets)
            if anim == "sweep":
                # 등치값을 낮은→높은 |field|로 자동 스윕. 프레임마다 메시 지오메트리
                # 전체를 교체(각 등치면 수천 점이라 경량). 등치값별 지오메트리는 1회만
                # 계산해 타일끼리 재사용한다.
                # v11 항목4: 슬라이더가 0 m/s ~ 최대값 전체 범위를 나타내도록
                # 선형 레벨(레벨 수는 v10 의 3배 유지). 0 등치면은 벽면 자체라
                # 빈 지오메트리(STL 만 표시)일 수 있으며 이는 물리적으로 옳다.
                _nlv = max(2, min(int(n_frames) * 2, 48))
                _lvls = np.unique(np.round(
                    np.linspace(lo_disp, vmax, _nlv), 6))
                _geoms = {float(lv): _contour_geom(lv) for lv in _lvls}
                # v11: 초기 표시는 중간 레벨 — 0 등치면(빈 지오메트리)으로 시작하면
                # 화면이 비어 보인다. 슬라이더 active 도 동일 인덱스로 동기화.
                _init_idx = len(_lvls) // 2
                lv0 = float(_lvls[_init_idx])
                first = True
                for (ox, oy) in offsets:
                    fig.add_trace(_mesh(_geoms[lv0], ox, oy, first)); first = False
                # 항목4: 스윕 모드에도 STL 형상을 함께 표시(스윕 중 고정).
                # 프레임은 앞쪽 n_tiles 트레이스만 교체하므로 경계는 유지된다.
                for t in self._boundary_traces(go, offsets, opacity=stl_opacity):
                    fig.add_trace(t)
                fig.frames = [
                    go.Frame(data=[_mesh(_geoms[float(lv)], ox, oy, (idx == 0))
                                   for idx, (ox, oy) in enumerate(offsets)],
                             traces=list(range(n_tiles)), name=f"{lv:.3g}")
                    for lv in _lvls]
            else:
                # 정적/회전: level 지정 시 그 등치면 1개, 없으면 3개 등치면 + 경계면
                if level is not None:
                    _lvls = [float(level)]
                else:
                    # v11 항목5: 0~최대 전체 범위의 3개 등치면(저속 영역 포함)
                    _lvls = [lo_disp + (vmax - lo_disp) * f
                             for f in (0.25, 0.5, 0.75)]
                first = True
                for lv in _lvls:
                    g = _contour_geom(lv)
                    for (ox, oy) in offsets:
                        fig.add_trace(_mesh(g, ox, oy, first)); first = False
                for t in self._boundary_traces(go, offsets, opacity=stl_opacity):
                    fig.add_trace(t)
                if anim == "rotate":
                    _nf = max(2, int(n_frames))
                    # fig.frames = [
                    #     go.Frame(layout=go.Layout(scene=dict(camera=dict(
                    #         eye=dict(x=1.7*math.cos(2*math.pi*fi/_nf),
                    #                  y=1.7*math.sin(2*math.pi*fi/_nf), z=1.0)))),
                    #         name=str(fi))
                    #     for fi in range(_nf)]
                    fig.frames = [
                        go.Frame(
                            name=str(fi)
                        )
                        for fi in range(_nf)
                    ]

            # v10 항목4: 유동방향 화살표(고정 트레이스 — 스윕 프레임은 앞쪽
            # n_tiles 트레이스만 교체하므로 영향 없음)
            for t in self._flow_arrow_traces(go, b):
                fig.add_trace(t)

            _menus = []
            if anim in ("rotate", "sweep"):
                _dur = 90 if anim == "rotate" else 140
                _menus = [dict(
                    type="buttons", showactive=False, direction="right",
                    x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#1a4a8a",
                    borderwidth=1, pad=dict(t=3, b=3, l=5, r=5),
                    font=dict(color="#10243e", size=12),
                    buttons=[
                        dict(label="▶ 재생", method="animate",
                             args=[None, dict(
                                mode="immediate", fromcurrent=True,
                                frame=dict(duration=_dur, redraw=True),
                                              transition=dict(duration=0))]),
                        dict(label="⏸ 정지", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode="immediate")]),
                    ])]

            _ann = "입체 등치면" + ({"rotate": " · 카메라 회전 재생",
                                   "sweep": " · 등치값 자동 스윕"}.get(anim, ""))
            # _layout_kw = dict(
            #     annotations=[dict(text=_ann, xref="paper", yref="paper",
            #         x=0.01, y=0.99, xanchor="left", yanchor="top", showarrow=False,
            #         font=dict(size=11, color="#1a4a8a"),
            #         bgcolor="rgba(255,255,255,0.82)", borderpad=4,
            #         bordercolor="#1a4a8a", borderwidth=1)],
                # scene=self._iso_scene(go, b, tile_nx, tile_ny, dx, dy,
                #                       init_camera=init_camera, anim=anim),
                
            _scene = self._iso_scene(
                go,
                b,
                tile_nx,
                tile_ny,
                dx,
                dy,
                init_camera=init_camera,
                anim=anim
            )

            _scene.pop("camera", None)

            _layout_kw = dict(
                annotations=[dict(text=_ann, xref="paper", yref="paper",
                    x=0.01, y=0.99, xanchor="left", yanchor="top", showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.92)", borderpad=4,
                    bordercolor="#333", borderwidth=1)],
                scene=_scene,
                updatemenus=_menus,
                showlegend=False,
                margin=dict(
                    l=0,
                    r=0,
                    t=10,
                    b=0
                ),  
                height=520,
                paper_bgcolor='#f0f8ff',
            )

            _layout_kw['uirevision'] = 'flowfield'

            if anim == 'sweep' and fig.frames:
                _iso_steps = [
                    dict(
                        method='animate',
                        args=[
                            [f.name], 
                            dict(
                                mode='immediate',
                                fromcurrent=True,
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0)
                            ),
                        ],                        
                        # v11: 레벨 48개 — 라벨은 1/8 만 표기(겹침 방지,
                        # currentvalue 에 정확값 상시 표시)
                        label=(f"{float(f.name):.3g}"
                               if (_fi % max(1, len(fig.frames)//8) == 0)
                               else ""),
                    )
                    for _fi, f in enumerate(fig.frames)
                ]
                _layout_kw['sliders'] = [dict(
                    active=_init_idx, pad=dict(b=10, t=10),
                    len=0.85, x=0.075, y=0,
                    # 항목3: 대비 강화 · 항목2: 단위 표시(등치값은 |field| → 필드 단위)
                    bgcolor="#1a4a8a", bordercolor="#10243e", borderwidth=1,
                    tickcolor="#10243e", tickwidth=1, font=dict(color="black", size=11),
                    currentvalue=dict(prefix='등치값: ', suffix=f' {unit}',
                                      visible=True, xanchor='right',
                                      font=dict(size=12, color="black")),
                    transition=dict(duration=0),
                    
                    # currentvalue=dict(
                    #     prefix='등치값: ',
                    #     visible=True,
                    #     xanchor='right',
                    #     font=dict(size=11)
                    # ),
                    
                    steps=_iso_steps,
                )]
                _layout_kw['margin'] = dict(l=0, r=0, t=10, b=60)
                _layout_kw['height'] = 580
            fig.update_layout(**_layout_kw)
            return fig
        except Exception as e:
            logger.error(f"render_field_3d 오류: {e}")
            return None

    def plot_residuals_plotly(self) -> Optional[Any]:
        """잔차 수렴 이력 Plotly 인터랙티브 그래프"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        residuals = self.reader.read_residuals()
        if not residuals:
            return None

        colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']
        # 솔버 필드명을 읽기 쉬운 범례 라벨로(없으면 원본 그대로)
        _label = {"Ux": "Ux (속도 x)", "Uy": "Uy (속도 y)", "Uz": "Uz (속도 z)",
                  "p": "p (압력)", "k": "k (난류에너지)", "omega": "ω (비소산율)",
                  "epsilon": "ε (소산율)", "nut": "νt"}
        fig = go.Figure()

        max_iter = 0
        for i, (fname, vals) in enumerate(residuals.items()):
            if not vals:
                continue
            max_iter = max(max_iter, len(vals))
            _nm = _label.get(fname, fname) or fname
            fig.add_trace(go.Scatter(
                x=list(range(1, len(vals)+1)),
                y=vals,
                mode='lines',
                name=_nm,
                line=dict(color=colors[i % len(colors)], width=1.8),
                hovertemplate=f"{_nm}: %{{y:.2e}}  iter %{{x}}<extra></extra>",
            ))

        if max_iter > 0:
            fig.add_trace(go.Scatter(
                x=[1, max_iter], y=[1e-4, 1e-4],
                mode='lines', name='수렴 목표 (1e-4)',
                line=dict(color='red', width=1, dash='dash'),
            ))

        # v13 항목10: 텍스트 선명도 개선 — plotly 는 텍스트를 SVG(벡터)로 그리므로
        # 확대해도 원리상 선명하다. 흐릿하게 보이던 원인은 (1) 작은 폰트, (2) 옅은
        # 색, (3) 폰트 패밀리 미지정으로 인한 렌더 편차. 시스템 산세리프를 명시하고
        # 폰트 크기·굵기·대비를 높여 모든 글자가 또렷하게 보이도록 한다.
        _FAM = "Arial, 'Helvetica Neue', Helvetica, sans-serif"
        _AXT = dict(family=_FAM, size=14, color="#111")   # 축 제목
        _TKF = dict(family=_FAM, size=12, color="#111")   # 눈금
        fig.update_layout(
            font=dict(family=_FAM, color="#111", size=13),
            xaxis=dict(title=dict(text="Iteration", font=_AXT),
                       tickfont=_TKF, gridcolor='#d9d9d9', showgrid=True,
                       linecolor="#888", ticks="outside", tickcolor="#888"),
            yaxis=dict(title=dict(text="Residual", font=_AXT), type='log',
                       tickfont=_TKF, gridcolor='#d9d9d9', showgrid=True,
                       exponentformat='e', linecolor="#888",
                       ticks="outside", tickcolor="#888"),
            title=dict(text="Convergence History",
                       font=dict(family=_FAM, size=17, color="#111"), x=0.5),
            showlegend=True,
            legend=dict(font=dict(family=_FAM, size=13, color="#111"),
                        title=dict(text="필드",
                                   font=dict(family=_FAM, size=13, color="#111")),
                        bgcolor='rgba(255,255,255,0.95)',
                        bordercolor='#888', borderwidth=1),
            hovermode='x unified',
            margin=dict(l=64, r=20, t=54, b=62),
            height=420,
            paper_bgcolor='white',
            plot_bgcolor='#ffffff',
        )
        return fig

    def plot_velocity_attenuation_plotly(self, u_inlet: float = 1.0) -> Optional[Any]:
        """유속 감쇠 프로파일 Plotly 인터랙티브 그래프 (subplot 2열)"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return None

        sampled = self.reader.read_sampled_data()
        if not sampled:
            return None

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Velocity |U| [m/s]", "Velocity Attenuation [%]"),
            horizontal_spacing=0.12,
        )
        colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']

        for i, (name, data) in enumerate(sampled.items()):
            if data is None or data.ndim < 2 or data.shape[0] < 2:
                continue
            try:
                if data.shape[1] >= 4:
                    pos   = data[:, 0]
                    u_mag = np.sqrt(data[:,1]**2 + data[:,2]**2 + data[:,3]**2)
                    clr   = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=pos, y=u_mag, mode='lines+markers',
                        name=name, line=dict(color=clr, width=1.5),
                        marker=dict(size=4),
                        hovertemplate="pos=%{x:.3f} m<br>|U|=%{y:.4f} m/s<extra></extra>",
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=pos, y=u_mag/u_inlet*100,
                        mode='lines+markers',
                        name=name, showlegend=False,
                        line=dict(color=clr, width=1.5, dash='dot'),
                        marker=dict(size=4),
                        hovertemplate="pos=%{x:.3f} m<br>att=%{y:.1f}%<extra></extra>",
                    ), row=1, col=2)
            except Exception:
                pass

        # 항목2: 배경이 흰색 고정이므로 모든 텍스트를 진한 색으로 고정해 어떤 테마·해상도
        # 에서도 또렷하게(테마 의존 기본색 회피, 저대비로 흐릿해 보이는 문제 해소).
        _dark = "#1a1a1a"
        fig.update_xaxes(title_text="Position [m]", gridcolor='#d0d0d0', showgrid=True,
                         title_font=dict(color=_dark, size=12),
                         tickfont=dict(color=_dark, size=10),
                         linecolor='#888', zerolinecolor='#bbb')
        fig.update_yaxes(gridcolor='#d0d0d0', showgrid=True,
                         title_font=dict(color=_dark, size=12),
                         tickfont=dict(color=_dark, size=10),
                         linecolor='#888', zerolinecolor='#bbb')
        fig.update_layout(
            title=dict(text="Velocity Profile (Wake)",
                       font=dict(size=15, color=_dark), x=0.5),
            height=420,
            font=dict(color=_dark),
            legend=dict(font=dict(size=11, color=_dark),
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='#888', borderwidth=1),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            margin=dict(l=60, r=20, t=60, b=60),
        )
        fig.update_annotations(font=dict(color=_dark, size=13))  # subplot 제목
        return fig

    def plot_force_coefficients_plotly(self, csv_path: Path) -> Optional[Any]:
        """Cd/Cl 계수 Plotly 인터랙티브 그래프"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import csv as csvmod
        except ImportError:
            return None

        if not csv_path.exists():
            return None

        try:
            rows = []
            with open(csv_path) as f:
                for row in csvmod.DictReader(f):
                    rows.append(row)
            if not rows:
                return None

            speeds = sorted(set(float(r["speed_m_s"]) for r in rows))
            colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Cd vs Angle of Attack",
                                                "Cl vs Angle of Attack"),
                                horizontal_spacing=0.12)

            for j, speed in enumerate(speeds):
                srows = [r for r in rows if abs(float(r["speed_m_s"]) - speed) < 0.01]
                clr   = colors[j % len(colors)]
                for col_idx, coeff in enumerate(["Cd", "Cl"], start=1):
                    ang_v = [float(r["angle_deg"]) for r in srows
                             if r.get(coeff, "") not in ("nan", "")]
                    val_v = [float(r[coeff]) for r in srows
                             if r.get(coeff, "") not in ("nan", "")]
                    if not val_v:
                        continue
                    fig.add_trace(go.Scatter(
                        x=ang_v, y=val_v, mode='lines+markers',
                        name=f"U={speed:.2f} m/s",
                        showlegend=(col_idx == 1),
                        line=dict(color=clr, width=1.8),
                        marker=dict(size=6),
                    ), row=1, col=col_idx)

            # 항목6: 배경이 흰색 고정이므로 모든 텍스트를 진한 색으로 고정해
            # Streamlit 다크 테마에서도 가독성을 보장(테마 의존 기본색 회피).
            _dark = "#1a1a1a"
            fig.update_xaxes(title_text="Angle of Attack [deg]",
                             gridcolor='#d0d0d0', showgrid=True,
                             title_font=dict(color=_dark, size=12),
                             tickfont=dict(color=_dark, size=10),
                             linecolor='#888', zerolinecolor='#bbb')
            fig.update_yaxes(gridcolor='#d0d0d0', showgrid=True,
                             title_font=dict(color=_dark, size=12),
                             tickfont=dict(color=_dark, size=10),
                             linecolor='#888', zerolinecolor='#bbb')
            fig.update_layout(
                title=dict(text="Force Coefficients (Cd / Cl)", x=0.5,
                           font=dict(size=14, color=_dark)),
                height=420, paper_bgcolor='white', plot_bgcolor='#fafafa',
                font=dict(color=_dark),
                legend=dict(font=dict(size=11, color=_dark),
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='#888', borderwidth=1),
                margin=dict(l=60, r=20, t=60, b=60),
            )
            # subplot 제목(annotations)도 진한 색으로
            fig.update_annotations(font=dict(color=_dark, size=13))
            return fig

        except Exception as e:
            logger.error(f"plot_force_coefficients_plotly 오류: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# 자동 갱신 시각화 워커
# ═══════════════════════════════════════════════════════════════════════════

class AutoRefreshVisualizer:
    """
    백그라운드 스레드로 주기적 자동 갱신 시각화
    Streamlit st.session_state와 연동
    """

    def __init__(self, case_dir: Path,
                 refresh_interval: float = 10.0,
                 on_update: Optional[callable] = None):
        self.case_dir         = Path(case_dir)
        self.refresh_interval = refresh_interval
        self.on_update        = on_update
        self._stop_flag       = threading.Event()
        self._thread          = None
        self.latest_images: Dict[str, str] = {}
        self.last_update_time: Optional[float] = None
        self.viz = CFDVisualizer(case_dir)

    def start(self, fields: List[str] = None):
        """자동 갱신 시작"""
        if fields is None:
            fields = ["U", "p"]
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(fields,),
            daemon=True
        )
        self._thread.start()
        logger.info(f"자동 갱신 시작: {self.refresh_interval}초 간격, 필드={fields}")

    def stop(self):
        """자동 갱신 중지"""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _worker(self, fields: List[str]):
        while not self._stop_flag.is_set():
            try:
                self.viz.invalidate_cache()
                updated = {}
                for field in fields:
                    img = self.viz.render_field(field)
                    if img:
                        updated[field] = img
                resid_img = self.viz.plot_residuals()
                if resid_img:
                    updated["residuals"] = resid_img

                self.latest_images.update(updated)
                self.last_update_time = time.time()

                if self.on_update:
                    self.on_update(self.latest_images)

            except Exception as e:
                logger.error(f"자동 갱신 오류: {e}")

            self._stop_flag.wait(timeout=self.refresh_interval)
