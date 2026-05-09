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
        """로그 파일에서 잔차 히스토리 파싱"""
        from pathlib import Path
        import re

        log_files = list(self.case_dir.glob("*.log")) + \
                    list(self.case_dir.glob("log.*"))
        if not log_files:
            return None

        # 가장 최신 로그 파일
        log_file = max(log_files, key=lambda f: f.stat().st_mtime)
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
