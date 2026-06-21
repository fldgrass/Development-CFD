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

    def __init__(self, case_dir: Path, window_size: Tuple[int,int] = (900, 600)):
        self.case_dir    = Path(case_dir)
        self.window_size = window_size
        self.reader      = OpenFOAMResultReader(case_dir)
        self._mesh_cache = None
        self._cache_time = 0

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
            self._mesh_cache = self.reader.load_openfoam_mesh()
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
                             show_streamlines: bool = False) -> Optional[Any]:
        """
        PyVista로 슬라이스 추출 → Plotly go.Mesh3d 인터랙티브 3D 뷰어 반환.
        slice_fraction: 0.0(경계 최소) ~ 1.0(경계 최대) 위치 비율.
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
            pos = lo + (hi - lo) * slice_fraction

            normal_map = {"x": (1,0,0), "y": (0,1,0), "z": (0,0,1)}
            normal_vec = normal_map.get(slice_normal, (0,1,0))
            cx = (bounds[0]+bounds[1])/2
            cy = (bounds[2]+bounds[3])/2
            cz = (bounds[4]+bounds[5])/2
            origin = {"x": (pos,cy,cz), "y": (cx,pos,cz), "z": (cx,cy,pos)}.get(
                slice_normal, (cx, pos, cz))

            sliced = internal.slice(normal=normal_vec, origin=origin)
            if sliced.n_points == 0:
                return None

            # 삼각화 후 메시 데이터 추출
            tri = sliced.triangulate()
            pts = tri.points
            raw_faces = tri.faces
            if len(raw_faces) == 0:
                return None
            faces = raw_faces.reshape(-1, 4)[:, 1:]

            # 스칼라 값 추출 (point data 우선)
            def _get_scalar(ds, fname):
                if fname not in ds.array_names:
                    return np.zeros(ds.n_points)
                arr = ds[fname]
                return np.linalg.norm(arr, axis=1) if arr.ndim == 2 else np.asarray(arr, float)

            scalar = _get_scalar(tri, field)

            cmap   = self._PLOTLY_CMAP.get(field, "Jet")
            unit   = self._FIELD_UNIT.get(field, "")
            cfg    = self.FIELD_CONFIG.get(field, self.FIELD_CONFIG["U"])

            fig = go.Figure()
            fig.add_trace(go.Mesh3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                i=faces[:,0], j=faces[:,1], k=faces[:,2],
                intensity=scalar,
                colorscale=cmap,
                colorbar=dict(
                    title=dict(text=f"{field} [{unit}]", side="right",
                               font=dict(size=12)),
                    thickness=14, len=0.75,
                    tickfont=dict(size=10),
                ),
                showscale=True,
                flatshading=False,
                lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
                showlegend=False,
                hovertemplate=f"{field}: %{{intensity:.4f}} {unit}<extra></extra>",
            ))

            # 경계면 (반투명 회색)
            for key in mesh.keys():
                if key == "internalMesh":
                    continue
                try:
                    patch = mesh[key]
                    if patch.n_points == 0:
                        continue
                    ptri = patch.triangulate()
                    pf   = ptri.faces.reshape(-1, 4)[:, 1:]
                    pp   = ptri.points
                    fig.add_trace(go.Mesh3d(
                        x=pp[:,0], y=pp[:,1], z=pp[:,2],
                        i=pf[:,0], j=pf[:,1], k=pf[:,2],
                        color='lightgray', opacity=0.15,
                        showscale=False, showlegend=False,
                        hoverinfo='skip',
                    ))
                except Exception:
                    pass

            # 유선 (Scatter3d 라인)
            if show_streamlines and PYVISTA_OK and "U" in internal.array_names:
                try:
                    seeds = pv.Sphere(radius=(hi-lo)*0.05, center=list(origin))
                    stream = internal.streamlines_from_source(
                        seeds, vectors="U", max_steps=500, max_step_length=0.05)
                    if stream.n_points > 0:
                        sp = stream.points
                        fig.add_trace(go.Scatter3d(
                            x=sp[:,0], y=sp[:,1], z=sp[:,2],
                            mode='lines',
                            line=dict(color='black', width=1),
                            showlegend=False, hoverinfo='skip',
                        ))
                except Exception:
                    pass

            # Annotation으로 슬라이스 위치 표시
            pct = int(slice_fraction * 100)
            fig.update_layout(
                annotations=[dict(
                    text=f"Slice {slice_normal.upper()} = {pos:.4f} m  ({pct}%)",
                    xref="paper", yref="paper",
                    x=0.01, y=0.99,
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=11, color="#1a4a8a"),
                    bgcolor="rgba(255,255,255,0.82)", borderpad=4,
                    bordercolor="#1a4a8a", borderwidth=1,
                )],
                scene=dict(
                    xaxis=dict(title="X [m]", backgroundcolor="#eaf4fb",
                               gridcolor="white", showbackground=True),
                    yaxis=dict(title="Y [m]", backgroundcolor="#eaf4fb",
                               gridcolor="white", showbackground=True),
                    zaxis=dict(title="Z [m]", backgroundcolor="#dce9f5",
                               gridcolor="white", showbackground=True),
                    aspectmode='data',
                    bgcolor='rgba(240,248,255,1)',
                    # 카메라를 약 2배 멀리 → 초기 화면에서 형상이 ~50% 작게 보임.
                    # (기존 eye(1.5,1.0,1.0) → (3.0,2.0,2.0)) 멀어서 스크롤로
                    # 줌아웃할 여유 공간도 약 2배로 늘어난다.
                    camera=dict(eye=dict(x=3.0, y=2.0, z=2.0)),
                ),
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=360,
                paper_bgcolor='#f0f8ff',
            )
            return fig

        except Exception as e:
            logger.error(f"render_field_plotly 오류: {e}")
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
        fig = go.Figure()

        max_iter = 0
        for i, (fname, vals) in enumerate(residuals.items()):
            if not vals:
                continue
            max_iter = max(max_iter, len(vals))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(vals)+1)),
                y=vals,
                mode='lines',
                name=fname,
                line=dict(color=colors[i % len(colors)], width=1.8),
                hovertemplate=f"{fname}: %{{y:.2e}}  iter %{{x}}<extra></extra>",
            ))

        if max_iter > 0:
            fig.add_trace(go.Scatter(
                x=[1, max_iter], y=[1e-4, 1e-4],
                mode='lines', name='Target (1e-4)',
                line=dict(color='red', width=1, dash='dash'),
            ))

        fig.update_layout(
            xaxis=dict(title="Iteration", gridcolor='lightgray', showgrid=True),
            yaxis=dict(title="Residual", type='log', gridcolor='lightgray',
                       showgrid=True, exponentformat='e'),
            title=dict(text="Convergence History", font=dict(size=14), x=0.5),
            legend=dict(font=dict(size=11), bgcolor='rgba(255,255,255,0.85)',
                        bordercolor='lightgray', borderwidth=1),
            hovermode='x unified',
            margin=dict(l=60, r=20, t=50, b=60),
            height=420,
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
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

        fig.update_xaxes(title_text="Position [m]", gridcolor='lightgray', showgrid=True)
        fig.update_yaxes(gridcolor='lightgray', showgrid=True)
        fig.update_layout(
            title=dict(text="Velocity Profile (Wake)", font=dict(size=14), x=0.5),
            height=420,
            legend=dict(font=dict(size=11)),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            margin=dict(l=60, r=20, t=60, b=60),
        )
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

            fig.update_xaxes(title_text="Angle of Attack [deg]",
                             gridcolor='lightgray', showgrid=True)
            fig.update_yaxes(gridcolor='lightgray', showgrid=True)
            fig.update_layout(
                title=dict(text="Force Coefficients (Cd / Cl)", x=0.5,
                           font=dict(size=14)),
                height=420, paper_bgcolor='white', plot_bgcolor='#fafafa',
                legend=dict(font=dict(size=11)),
                margin=dict(l=60, r=20, t=60, b=60),
            )
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
