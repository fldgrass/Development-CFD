"""
app.py  ─  양식 가두리 CFD 해석 시스템 메인 대시보드
=============================================================
OpenFOAM 기반 수산공학 CFD 자동화 시스템
Streamlit + PyVista 통합 GUI

실행: streamlit run app.py --server.port 8501
"""

import os
import sys
import time
import json
import math
import shutil
import threading
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import streamlit as st
from streamlit import session_state as ss

# ─── 경로 설정 ────────────────────────────────────────────────────────────
APP_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "modules"))

from cfd_manager import (
    UnitCellCaseBuilder, FullStructureCaseBuilder,
    OpenFOAMRunner, ResultExtractor, BatchAnalysisManager,
    get_cpu_count, RESULTS_DIR, STL_UPLOAD_DIR, LOGS_DIR, BASE_DIR
)
from visualizer import CFDVisualizer, AutoRefreshVisualizer, OpenFOAMResultReader

# ═══════════════════════════════════════════════════════════════════════════
#  Streamlit 전역 설정
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="양식 가두리 CFD 해석 시스템",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://openfoam.org/documentation/",
        "Report a bug": None,
        "About": "수산공학 전공 OpenFOAM CFD 자동화 시스템 v1.0",
    }
)

# ─── 커스텀 CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* 전체 폰트 */
  html, body, [class*="css"] { font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; }

  /* 사이드바 */
  .css-1d391kg { background-color: #0a2342; }
  section[data-testid="stSidebar"] { background: linear-gradient(160deg,#0d2b5e,#1a4a8a); }
  section[data-testid="stSidebar"] * { color: #e8f4fd !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label { color: #a8d4f5 !important; }

  /* 상태 뱃지 */
  .badge-running  { background:#1a73e8; color:white; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }
  .badge-done     { background:#0f9d58; color:white; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }
  .badge-error    { background:#d93025; color:white; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }
  .badge-idle     { background:#5f6368; color:white; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }

  /* 메트릭 카드 */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f8fbff, #e8f4fd);
    border: 1px solid #c5dff8; border-radius: 10px; padding: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
  }

  /* 버튼 */
  .stButton > button {
    border-radius: 8px; font-weight: 600; transition: all 0.2s;
  }
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

  /* 헤더 */
  h1 { color: #0d2b5e !important; border-bottom: 3px solid #1a73e8; padding-bottom: 8px; }
  h2 { color: #1a4a8a !important; }
  h3 { color: #1a73e8 !important; }

  /* 로그 박스 */
  .log-box {
    background: #0d1117; color: #58d68d; font-family: monospace;
    font-size: 12px; padding: 12px; border-radius: 8px;
    height: 280px; overflow-y: auto; white-space: pre-wrap;
    border: 1px solid #30363d;
  }

  /* 탭 */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 8px 18px;
    background: #e8f4fd; color: #1a4a8a; font-weight: 600;
  }
  .stTabs [aria-selected="true"] { background: #1a73e8 !important; color: white !important; }

  /* 파일 업로더 */
  .stFileUploader { border: 2px dashed #1a73e8; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  세션 상태 초기화
# ═══════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "analysis_mode":      "unit_cell",
        "job_status":         "idle",       # idle / running / done / error
        "job_thread":         None,
        "job_runner":         None,
        "batch_manager":      None,
        "progress":           0,
        "current_step":       "",
        "log_lines":          [],
        "last_result_dir":    None,
        "auto_viz":           None,
        "viz_images":         {},
        "results_csv":        None,
        "refresh_interval":   10,
        "n_cores":            get_cpu_count(),
        "stl_net_path":       None,
        "stl_cage_path":      None,
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

init_session()


# ═══════════════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════

def add_log(msg: str):
    ss.log_lines.append(f"[{datetime.now():%H:%M:%S}] {msg}")
    if len(ss.log_lines) > 500:
        ss.log_lines = ss.log_lines[-400:]

def set_status(status: str, step: str = ""):
    ss.job_status   = status
    ss.current_step = step

def status_badge(status: str) -> str:
    labels = {"idle":"대기","running":"해석 중","done":"완료","error":"오류"}
    return f'<span class="badge-{status}">{labels.get(status, status)}</span>'

def save_uploaded_stl(uploaded_file, prefix: str) -> Optional[Path]:
    """업로드된 STL 파일을 임시 디렉토리에 저장"""
    if uploaded_file is None:
        return None
    save_path = STL_UPLOAD_DIR / f"{prefix}_{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def check_openfoam() -> bool:
    """OpenFOAM 설치 여부 확인"""
    for cmd in ["blockMesh", "simpleFoam", "snappyHexMesh"]:
        if shutil.which(cmd):
            return True
    # bashrc 방식 확인
    bashrc = OpenFOAMRunner.find_openfoam_bashrc()
    return bashrc is not None

def format_log_html(lines: List[str]) -> str:
    """로그 텍스트를 HTML 로그 박스 포맷으로 변환"""
    colored = []
    for line in lines[-100:]:
        if "✅" in line or "완료" in line:
            colored.append(f'<span style="color:#58d68d">{line}</span>')
        elif "❌" in line or "오류" in line or "Error" in line:
            colored.append(f'<span style="color:#e74c3c">{line}</span>')
        elif "▶" in line or "시작" in line:
            colored.append(f'<span style="color:#5dade2">{line}</span>')
        elif "Time =" in line:
            colored.append(f'<span style="color:#f39c12">{line}</span>')
        elif "residual" in line.lower():
            colored.append(f'<span style="color:#a29bfe">{line}</span>')
        else:
            colored.append(line)
    return "\n".join(colored)


# ═══════════════════════════════════════════════════════════════════════════
#  사이드바
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌊 CFD 해석 시스템")
    st.markdown("**수산공학 양식 가두리 해석**")
    st.divider()

    # ─── 해석 모드 선택 ───────────────────────────────────────────────────
    st.markdown("### ⚙️ 해석 모드")
    mode = st.radio(
        "모드 선택",
        options=["unit_cell", "full_structure"],
        format_func=lambda x: {
            "unit_cell":      "🔬 단위 셀 모드 (Cd/Cl DB 추출)",
            "full_structure": "🏗️ 전체 구조 모드 (가두리 유동장)",
        }[x],
        key="analysis_mode",
    )
    st.divider()

    # ─── 공통 물리 조건 ───────────────────────────────────────────────────
    st.markdown("### 🌊 물리 조건")
    rho = st.number_input("해수 밀도 ρ [kg/m³]", value=1025.0,
                          min_value=1000.0, max_value=1100.0, step=1.0)
    nu  = st.number_input("동점성계수 ν [×10⁻⁶ m²/s]",
                          value=1.19, min_value=0.5, max_value=2.0, step=0.01)
    ti  = st.slider("난류 강도 I [%]", 1, 20, 5)
    st.divider()

    # ─── 시스템 설정 ──────────────────────────────────────────────────────
    st.markdown("### 🖥️ 시스템 설정")
    max_cores = get_cpu_count()
    n_cores = st.slider("MPI 코어 수", 1, max(max_cores, 32), max_cores,
                        help=f"감지된 코어 수: {max_cores}")
    ss.n_cores = n_cores

    refresh_sec = st.slider("시각화 갱신 주기 [초]", 5, 60, 10)
    ss.refresh_interval = refresh_sec
    st.divider()

    # ─── OpenFOAM 상태 ────────────────────────────────────────────────────
    st.markdown("### 🔍 시스템 상태")
    of_ok = check_openfoam()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("OpenFOAM", "✅ 설치됨" if of_ok else "❌ 미설치")
    with col2:
        st.metric("MPI 코어", str(n_cores))

    try:
        import pyvista as pv
        st.success("PyVista ✅")
    except ImportError:
        st.warning("PyVista ⚠️ (설치 권장)")

    if not of_ok:
        st.error("⚠️ OpenFOAM이 감지되지 않았습니다.\n설치 후 재실행하세요.")


# ═══════════════════════════════════════════════════════════════════════════
#  메인 화면
# ═══════════════════════════════════════════════════════════════════════════

# ─── 헤더 ────────────────────────────────────────────────────────────────
st.markdown("# 🌊 양식 가두리 CFD 해석 시스템")
st.markdown(
    f"**해석 모드:** {'🔬 단위 셀 (Unit Cell)' if mode=='unit_cell' else '🏗️ 전체 구조 (Full Structure)'}  "
    f"&nbsp;&nbsp;**상태:** {status_badge(ss.job_status)}  "
    f"&nbsp;&nbsp;**진행률:** {ss.progress}%",
    unsafe_allow_html=True
)

# ─── 진행률 바 ────────────────────────────────────────────────────────────
if ss.job_status == "running":
    st.progress(ss.progress / 100, text=f"⚙️ {ss.current_step}")
elif ss.job_status == "done":
    st.progress(1.0, text="✅ 해석 완료!")
elif ss.job_status == "error":
    st.error(f"❌ 오류 발생: {ss.current_step}")

st.divider()

# ─── 탭 구성 ─────────────────────────────────────────────────────────────
tab_input, tab_batch, tab_monitor, tab_results, tab_help = st.tabs([
    "📂 입력 설정",
    "🔄 배치 해석",
    "📊 실시간 모니터",
    "📈 결과 분석",
    "📖 도움말",
])


# ═══════════════════════════════════════════════════════════════════════════
#  탭 1: 입력 설정
# ═══════════════════════════════════════════════════════════════════════════

with tab_input:
    col_left, col_right = st.columns([1.2, 0.8], gap="large")

    with col_left:
        # ─── STL 파일 업로드 ──────────────────────────────────────────────
        st.markdown("### 📁 STL 파일 업로드")

        if mode == "unit_cell":
            st.info("💡 라이노3D에서 내보낸 **그물 단위 셀 STL** (매듭+그물발 포함)을 업로드하세요.")
            net_file = st.file_uploader(
                "그물 단위 셀 STL", type=["stl"],
                key="net_stl_upload",
                help="Rhino3D에서 1개 단위 셀(노드+트와인) STL 파일을 업로드"
            )
            if net_file:
                ss.stl_net_path = save_uploaded_stl(net_file, "net")
                st.success(f"✅ 업로드 완료: {net_file.name} ({net_file.size/1024:.1f} KB)")

                # STL 정보 표시
                with st.expander("🔍 STL 파일 정보"):
                    st.code(f"""
파일명   : {net_file.name}
크기     : {net_file.size/1024:.1f} KB
저장 경로: {ss.stl_net_path}
                    """)

        else:  # full_structure
            st.info("💡 **가두리 림(Rim) STL**과 **그물 STL**을 각각 업로드하세요.")
            cage_file = st.file_uploader(
                "🏗️ 가두리 림(Cage Rim) STL", type=["stl"],
                key="cage_stl_upload",
                help="원통형 가두리의 금속 림/프레임 STL"
            )
            net_file2 = st.file_uploader(
                "🕸️ 그물(Net) STL", type=["stl"],
                key="net_stl_upload2",
                help="원통형 가두리의 그물망 STL"
            )
            if cage_file:
                ss.stl_cage_path = save_uploaded_stl(cage_file, "cage")
                st.success(f"✅ 가두리 업로드: {cage_file.name}")
            if net_file2:
                ss.stl_net_path = save_uploaded_stl(net_file2, "net2")
                st.success(f"✅ 그물 업로드: {net_file2.name}")

        st.divider()

        # ─── 해석 파라미터 ────────────────────────────────────────────────
        st.markdown("### 🎛️ 해석 파라미터")

        if mode == "unit_cell":
            col_a, col_b = st.columns(2)
            with col_a:
                speed_val = st.number_input(
                    "유속 U [m/s]", value=1.0,
                    min_value=0.1, max_value=5.0, step=0.1,
                    key="speed_single"
                )
                cell_size = st.number_input(
                    "단위 셀 크기 a [mm]", value=20.0,
                    min_value=5.0, max_value=100.0, step=1.0,
                    key="cell_size_mm"
                ) / 1000.0

            with col_b:
                angle_val = st.number_input(
                    "영각 AoA [°]", value=0.0,
                    min_value=-90.0, max_value=90.0, step=5.0,
                    key="angle_single"
                )
                end_time = st.number_input(
                    "최대 반복 횟수", value=2000,
                    min_value=500, max_value=10000, step=500
                )

            # 속도 벡터 미리보기
            from cfd_manager import compute_velocity_vector
            Ux, Uy, Uz = compute_velocity_vector(speed_val, angle_val)
            st.info(
                f"📐 **속도 벡터** = ({Ux:.4f}, {Uy:.4f}, {Uz:.4f}) m/s  "
                f"| Reynolds = {speed_val * cell_size / 1.19e-6:.1f}"
            )

        else:  # full_structure
            col_a, col_b = st.columns(2)
            with col_a:
                speed_val = st.number_input(
                    "유속 U [m/s]", value=1.0,
                    min_value=0.1, max_value=5.0, step=0.1
                )
                cage_d = st.number_input(
                    "가두리 직경 D [m]", value=10.0,
                    min_value=1.0, max_value=50.0, step=0.5
                )
            with col_b:
                angle_val = st.number_input(
                    "유입 영각 [°]", value=0.0,
                    min_value=-45.0, max_value=45.0, step=5.0
                )
                cage_h = st.number_input(
                    "가두리 수심 H [m]", value=5.0,
                    min_value=1.0, max_value=30.0, step=0.5
                )
            end_time = st.number_input(
                "최대 반복 횟수", value=3000,
                min_value=500, max_value=15000, step=500
            )
            Re = speed_val * cage_d / 1.19e-6
            st.info(f"📐 **가두리 Reynolds 수** = {Re:.2e}  |  도메인: {3*cage_d:.0f}D × {3*cage_d:.0f}D × {cage_h:.0f}m")

    with col_right:
        # ─── 해석 정보 요약 ───────────────────────────────────────────────
        st.markdown("### 📋 해석 설정 요약")

        summary_data = {
            "해석 모드":    "단위 셀" if mode == "unit_cell" else "전체 구조",
            "유속":         f"{speed_val:.2f} m/s",
            "영각":         f"{angle_val:.1f}°",
            "MPI 코어":     f"{n_cores}개",
            "최대 반복":    str(end_time),
            "해수 밀도":    f"{rho:.1f} kg/m³",
            "동점성계수":   f"{nu:.2f} × 10⁻⁶ m²/s",
            "난류 강도":    f"{ti}%",
        }
        if mode == "unit_cell":
            summary_data["단위 셀 크기"] = f"{cell_size*1000:.1f} mm"
        else:
            summary_data["가두리 직경"] = f"{cage_d:.1f} m"
            summary_data["가두리 수심"] = f"{cage_h:.1f} m"

        for key, val in summary_data.items():
            col_k, col_v = st.columns([1, 1])
            col_k.markdown(f"**{key}**")
            col_v.markdown(val)

        st.divider()

        # ─── 난류 파라미터 자동 계산 ──────────────────────────────────────
        from cfd_manager import compute_turbulence_params
        L = cell_size if mode == "unit_cell" else cage_d * 0.1
        turb = compute_turbulence_params(speed_val, ti/100, L)
        st.markdown("### 🌀 난류 초기조건 (자동 계산)")
        col_k1, col_k2 = st.columns(2)
        col_k1.metric("k [m²/s²]", f"{turb['k']:.4e}")
        col_k2.metric("ω [1/s]",   f"{turb['omega']:.3f}")

        st.divider()

        # ─── 실행 버튼 ────────────────────────────────────────────────────
        st.markdown("### 🚀 해석 실행")

        run_disabled = (ss.job_status == "running")

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button(
                "▶️ 해석 시작",
                disabled=run_disabled,
                use_container_width=True,
                type="primary"
            ):
                _start_single_analysis(
                    mode, speed_val, angle_val,
                    cell_size if mode == "unit_cell" else None,
                    cage_d if mode == "full_structure" else None,
                    cage_h if mode == "full_structure" else None,
                    end_time, n_cores, rho, ti
                )

        with btn_col2:
            if st.button(
                "⏹️ 해석 중지",
                disabled=(ss.job_status != "running"),
                use_container_width=True,
                type="secondary"
            ):
                _stop_analysis()


# ═══════════════════════════════════════════════════════════════════════════
#  탭 2: 배치 해석
# ═══════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown("### 🔄 배치 해석 설정 (영각 × 유속 자동 순환)")
    st.info(
        "💡 영각과 유속 범위를 설정하면 모든 조합을 자동으로 순환 해석하여 "
        "Cd/Cl 데이터베이스 CSV를 자동 생성합니다."
    )

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### 유속 설정")
        u_min   = st.number_input("최소 유속 [m/s]", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
        u_max   = st.number_input("최대 유속 [m/s]", value=2.0, min_value=0.1, max_value=5.0, step=0.1)
        u_steps = st.number_input("유속 단계 수",    value=4,   min_value=1,   max_value=20,  step=1)
        speeds  = list(np.linspace(u_min, u_max, int(u_steps)))

    with col_b2:
        st.markdown("#### 영각 설정")
        a_min   = st.number_input("최소 영각 [°]", value=0.0,  min_value=-90.0, max_value=90.0, step=5.0)
        a_max   = st.number_input("최대 영각 [°]", value=45.0, min_value=-90.0, max_value=90.0, step=5.0)
        a_steps = st.number_input("영각 단계 수",  value=5,    min_value=1,     max_value=20,   step=1)
        angles  = list(np.linspace(a_min, a_max, int(a_steps)))

    # ─── 파라미터 매트릭스 미리보기 ──────────────────────────────────────
    st.markdown("#### 📊 해석 매트릭스 미리보기")
    total_cases = len(speeds) * len(angles)
    st.metric("총 해석 케이스", f"{total_cases}개",
              help="유속 × 영각 조합의 총 수")

    # 테이블 형식으로 표시
    import pandas as pd
    matrix_data = {}
    for s in speeds:
        matrix_data[f"U={s:.2f}"] = [f"U={s:.2f}, α={a:.1f}°" for a in angles]
    matrix_df = pd.DataFrame(
        matrix_data,
        index=[f"α={a:.1f}°" for a in angles]
    )
    st.dataframe(matrix_df, use_container_width=True)

    st.divider()

    # ─── CSV 저장 경로 ────────────────────────────────────────────────────
    csv_name = st.text_input(
        "결과 CSV 파일명",
        value=f"force_coeffs_{mode}_{datetime.now():%Y%m%d}.csv"
    )
    csv_path = RESULTS_DIR / mode / csv_name

    col_bb1, col_bb2 = st.columns(2)

    with col_bb1:
        batch_disabled = (ss.job_status == "running")
        if st.button(
            f"🚀 배치 해석 시작 ({total_cases}개 케이스)",
            disabled=batch_disabled,
            use_container_width=True,
            type="primary"
        ):
            _start_batch_analysis(
                mode, speeds, angles, csv_path, n_cores, rho, ti
            )

    with col_bb2:
        if st.button(
            "⏹️ 배치 중지",
            disabled=(ss.job_status != "running"),
            use_container_width=True
        ):
            _stop_analysis()

    # ─── 기존 결과 CSV 다운로드 ───────────────────────────────────────────
    st.divider()
    st.markdown("#### 📥 결과 파일 다운로드")
    result_csvs = list((RESULTS_DIR / mode).glob("*.csv"))
    if result_csvs:
        for csv_file in sorted(result_csvs, key=lambda f: f.stat().st_mtime, reverse=True):
            col_f1, col_f2, col_f3 = st.columns([3, 1, 1])
            col_f1.markdown(f"📄 **{csv_file.name}**")
            col_f2.markdown(f"_{csv_file.stat().st_size/1024:.1f} KB_")
            with open(csv_file, "rb") as f:
                col_f3.download_button(
                    "⬇️ 다운로드",
                    data=f.read(),
                    file_name=csv_file.name,
                    mime="text/csv",
                    key=f"dl_{csv_file.name}"
                )
    else:
        st.info("아직 저장된 결과 CSV가 없습니다. 배치 해석을 실행하세요.")


# ═══════════════════════════════════════════════════════════════════════════
#  탭 3: 실시간 모니터
# ═══════════════════════════════════════════════════════════════════════════

with tab_monitor:
    st.markdown("### 📊 실시간 해석 모니터링")

    # ─── 상단 메트릭 ──────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("해석 상태",   {"idle":"대기","running":"진행 중","done":"완료","error":"오류"}[ss.job_status])
    mc2.metric("진행률",      f"{ss.progress}%")
    mc3.metric("현재 단계",   ss.current_step or "-")
    mc4.metric("MPI 코어",    f"{n_cores}개 / GPU RTX3090 ×2")

    st.divider()

    # ─── 시각화 및 로그 ───────────────────────────────────────────────────
    viz_col, log_col = st.columns([1.4, 0.6])

    with viz_col:
        st.markdown("#### 🖼️ 3D 유동장 시각화")

        # 필드 선택
        field_options = {"U": "속도장 |U|", "p": "압력장", "k": "난류 운동에너지",
                         "omega": "비소산율", "residuals": "수렴 이력"}
        selected_field = st.selectbox(
            "시각화 필드",
            options=list(field_options.keys()),
            format_func=lambda x: field_options[x]
        )

        viz_subcol1, viz_subcol2 = st.columns(2)
        with viz_subcol1:
            slice_dir = st.selectbox("슬라이스 방향", ["y", "x", "z"])
        with viz_subcol2:
            show_stream = st.checkbox("유선 표시", value=False)

        # 이미지 표시
        if ss.last_result_dir and Path(ss.last_result_dir).exists():
            viz = CFDVisualizer(Path(ss.last_result_dir))

            if selected_field == "residuals":
                img_path = viz.plot_residuals()
            else:
                img_path = viz.render_field(
                    selected_field, slice_dir,
                    show_streamlines=show_stream
                )

            if img_path and Path(img_path).exists():
                st.image(img_path, use_container_width=True,
                         caption=f"{field_options[selected_field]} — {Path(ss.last_result_dir).name}")
            else:
                st.info("🔄 해석 결과를 기다리는 중입니다...")
                # 플레이스홀더 이미지
                st.markdown("""
                <div style="background:#f0f8ff; border:2px dashed #1a73e8;
                            border-radius:10px; padding:60px; text-align:center;">
                    <h3 style="color:#1a73e8;">🌊 해석 대기 중</h3>
                    <p style="color:#666;">STL 파일을 업로드하고 해석을 시작하면<br>
                    실시간으로 결과가 표시됩니다.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f0f8ff; border:2px dashed #1a73e8;
                        border-radius:10px; padding:60px; text-align:center;">
                <h3 style="color:#1a73e8;">🌊 해석 시작 전</h3>
                <p style="color:#666;">'입력 설정' 탭에서 STL 파일 업로드 후<br>
                해석을 시작하세요.</p>
            </div>
            """, unsafe_allow_html=True)

        # 자동 갱신 컨트롤
        if ss.job_status == "running":
            if st.button(f"🔄 수동 갱신 ({refresh_sec}초 자동 갱신 중)"):
                st.rerun()

    with log_col:
        st.markdown("#### 📟 해석 로그")
        log_html = format_log_html(ss.log_lines)
        st.markdown(
            f'<div class="log-box">{log_html}</div>',
            unsafe_allow_html=True
        )

        # 잔차 실시간 표시
        if ss.last_result_dir and ss.job_status == "running":
            runner = ss.get("job_runner")
            if runner and hasattr(runner, "last_residuals"):
                st.markdown("#### 📉 현재 잔차")
                for field, resid in runner.last_residuals.items():
                    color = "green" if resid < 1e-4 else "orange" if resid < 1e-3 else "red"
                    st.markdown(
                        f"**{field}**: "
                        f'<span style="color:{color}">{resid:.2e}</span>',
                        unsafe_allow_html=True
                    )

    # 자동 새로고침
    if ss.job_status == "running":
        time.sleep(refresh_sec)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  탭 4: 결과 분석
# ═══════════════════════════════════════════════════════════════════════════

with tab_results:
    st.markdown("### 📈 해석 결과 분석")

    # ─── 결과 디렉토리 선택 ───────────────────────────────────────────────
    all_cases = []
    for subdir in ["unit_cell", "full_structure"]:
        case_root = RESULTS_DIR / subdir
        if case_root.exists():
            all_cases.extend(
                [(subdir, d) for d in case_root.iterdir()
                 if d.is_dir() and (d / "constant").exists()]
            )

    if not all_cases:
        st.info("아직 해석된 케이스가 없습니다. '입력 설정' 탭에서 해석을 시작하세요.")
    else:
        case_options = {f"{s}/{d.name}": d for s, d in all_cases}
        selected_case_key = st.selectbox(
            "결과 케이스 선택",
            options=list(case_options.keys())
        )
        selected_case_dir = case_options[selected_case_key]

        # ─── 시각화 탭 ────────────────────────────────────────────────────
        r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
            "🌊 유동장", "📉 수렴 이력", "📊 유속 감쇠", "💾 CSV 데이터"
        ])

        with r_tab1:
            viz2 = CFDVisualizer(selected_case_dir)
            r_col1, r_col2 = st.columns([1, 1])
            with r_col1:
                st.markdown("**속도장 |U|**")
                img = viz2.render_field("U", "y")
                if img:
                    st.image(img, use_container_width=True)
            with r_col2:
                st.markdown("**압력장 p**")
                img2 = viz2.render_field("p", "y")
                if img2:
                    st.image(img2, use_container_width=True)

        with r_tab2:
            viz3 = CFDVisualizer(selected_case_dir)
            resid_img = viz3.plot_residuals()
            if resid_img:
                st.image(resid_img, use_container_width=True)
            else:
                st.info("수렴 이력 데이터가 없습니다. 해석 로그 파일을 확인하세요.")

        with r_tab3:
            viz4 = CFDVisualizer(selected_case_dir)
            atten_img = viz4.plot_velocity_attenuation()
            if atten_img:
                st.image(atten_img, use_container_width=True)
            else:
                st.info("유속 샘플링 데이터가 없습니다. (전체 구조 모드에서 이용 가능)")

        with r_tab4:
            # CSV 데이터 표시
            all_csvs = list((RESULTS_DIR / "unit_cell").glob("*.csv")) + \
                       list((RESULTS_DIR / "full_structure").glob("*.csv"))

            if all_csvs:
                csv_sel = st.selectbox(
                    "CSV 파일 선택",
                    options=[c.name for c in all_csvs],
                    key="result_csv_sel"
                )
                csv_target = next(c for c in all_csvs if c.name == csv_sel)

                try:
                    df = pd.read_csv(csv_target)
                    st.dataframe(df, use_container_width=True)

                    # Cd/Cl 플롯
                    viz5 = CFDVisualizer(selected_case_dir)
                    coeff_img = viz5.plot_force_coefficients(csv_target)
                    if coeff_img:
                        st.image(coeff_img, use_container_width=True)

                    # 다운로드
                    with open(csv_target, "rb") as f:
                        st.download_button(
                            "⬇️ CSV 다운로드 (질량-스프링 모델 호환 포맷)",
                            data=f.read(),
                            file_name=csv_target.name,
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"CSV 읽기 오류: {e}")
            else:
                st.info("저장된 결과 CSV가 없습니다.")


# ═══════════════════════════════════════════════════════════════════════════
#  탭 5: 도움말
# ═══════════════════════════════════════════════════════════════════════════

with tab_help:
    st.markdown("""
### 📖 시스템 사용 가이드

---

#### 🔬 단위 셀 모드 (Unit Cell Mode)
1. **STL 준비**: 라이노3D에서 그물 1개 단위 셀(매듭+그물발) STL 내보내기
2. **파일 업로드**: '입력 설정' 탭에서 STL 업로드
3. **조건 입력**: 유속, 영각, 단위 셀 크기 입력
4. **해석 실행**: '▶️ 해석 시작' 클릭
5. **결과 확인**: '결과 분석' 탭에서 Cd/Cl 값 확인 및 CSV 다운로드

**주기 경계조건(Cyclic)**: xMin↔xMax, yMin↔yMax 면이 자동으로 주기 조건으로 설정됩니다.

---

#### 🏗️ 전체 구조 모드 (Full Structure Mode)
1. **STL 준비**: 가두리 림 STL + 그물 STL 각각 내보내기
2. **파일 업로드**: 두 STL 파일 모두 업로드
3. **가두리 크기 입력**: 직경(D), 수심(H) 입력
4. **해석 실행**: 입구/출구/벽면 경계조건이 자동 설정됩니다
5. **결과 확인**: 유속 감쇠 분포, 항력 계수 확인

---

#### 🔄 배치 해석
- **배치 해석** 탭에서 유속/영각 범위 지정
- 모든 조합이 자동 실행되어 **CSV DB 자동 생성**
- 생성된 CSV는 **질량-스프링 모델(C++)**과 호환되는 표준 포맷

---

#### 🖥️ GPU 가속 (RTX 3090 × 2)
```
GAMG 솔버 → 선형 방정식 풀이
AmgX 플러그인 설치 시 → GPU 가속 자동 활성화
MPI 병렬화 → CPU 코어 수에 맞게 자동 설정
```

---

#### 📁 CSV 출력 포맷 (질량-스프링 모델 호환)
| 컬럼 | 설명 |
|------|------|
| speed_m_s | 유속 [m/s] |
| angle_deg | 영각 [°] |
| Cd | 항력 계수 |
| Cl | 양력 계수 |
| Cm | 모멘트 계수 |
| Fx_N, Fy_N, Fz_N | 각 방향 힘 [N] |
| rho_kg_m3 | 해수 밀도 [kg/m³] |

---

#### ⚠️ 문제 해결
- **OpenFOAM 미감지**: `/opt/openfoam*/etc/bashrc` 경로 확인
- **MPI 오류**: `sudo apt install openmpi-bin` 실행
- **격자 생성 실패**: STL 파일 법선 방향 확인 (`Rhino > Analyze > Direction`)
- **수렴 불안정**: relaxationFactors를 U:0.5, p:0.2로 줄이기
""")


# ═══════════════════════════════════════════════════════════════════════════
#  백엔드 함수 (버튼 핸들러)
# ═══════════════════════════════════════════════════════════════════════════

def _start_single_analysis(
    mode, speed, angle,
    cell_size, cage_d, cage_h,
    end_time, n_cores, rho, ti
):
    """단일 해석 케이스 실행 (백그라운드 스레드)"""
    if ss.job_status == "running":
        st.warning("이미 해석이 실행 중입니다.")
        return

    if not check_openfoam():
        st.error("❌ OpenFOAM이 설치되지 않았습니다. 설치 후 다시 시도하세요.")
        return

    stl_net  = ss.stl_net_path
    stl_cage = ss.stl_cage_path

    if mode == "unit_cell" and not stl_net:
        st.error("❌ 그물 STL 파일을 먼저 업로드하세요.")
        return

    set_status("running", "케이스 준비 중...")
    ss.progress   = 0
    ss.log_lines  = []
    add_log(f"해석 시작: 모드={mode}, U={speed}m/s, α={angle}°")

    # 케이스 디렉토리 생성
    case_name = f"{mode}_U{speed:.2f}_A{angle:.1f}_{datetime.now():%H%M%S}"
    case_dir  = RESULTS_DIR / mode / case_name
    ss.last_result_dir = str(case_dir)

    def _run():
        try:
            # 케이스 빌드
            if mode == "unit_cell":
                builder = UnitCellCaseBuilder(
                    case_dir=case_dir,
                    stl_path=stl_net,
                    speed=speed, angle_deg=angle,
                    cell_size=cell_size or 0.02,
                    n_cores=n_cores
                )
            else:
                builder = FullStructureCaseBuilder(
                    case_dir=case_dir,
                    cage_stl=stl_cage,
                    net_stl=stl_net,
                    speed=speed, angle_deg=angle,
                    cage_diameter=cage_d or 10.0,
                    cage_depth=cage_h or 5.0,
                    n_cores=n_cores
                )
            builder.build()
            add_log("✅ 케이스 빌드 완료")

            # 해석 실행
            runner = OpenFOAMRunner(
                case_dir=case_dir,
                n_cores=n_cores,
                progress_cb=lambda p, s, e: (
                    setattr(ss, "progress", p),
                    setattr(ss, "current_step", f"simpleFoam: {s}/{e}")
                ),
                log_cb=add_log
            )
            ss.job_runner = runner

            runner.run_blockMesh()
            set_status("running", "snappyHexMesh 실행 중...")
            runner.run_surfaceFeatureExtract()
            runner.run_snappyHexMesh()
            set_status("running", "CFD 해석 중...")
            runner.run_solver(end_time=end_time)
            runner.run_reconstructPar()

            # 결과 추출
            extractor = ResultExtractor(case_dir, speed, angle, rho)
            csv_out   = RESULTS_DIR / mode / f"results_{mode}.csv"
            extractor.save_csv(csv_out)
            ss.results_csv = str(csv_out)

            set_status("done", "해석 완료!")
            add_log(f"✅ 해석 완료! 결과: {csv_out}")

        except Exception as e:
            set_status("error", str(e))
            add_log(f"❌ 오류: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    ss.job_thread = thread
    thread.start()
    st.rerun()


def _start_batch_analysis(mode, speeds, angles, csv_path, n_cores, rho, ti):
    """배치 해석 실행 (백그라운드 스레드)"""
    if ss.job_status == "running":
        st.warning("이미 해석이 실행 중입니다.")
        return

    stl_paths = {}
    if ss.stl_net_path:
        stl_paths["net"] = Path(ss.stl_net_path)
    if ss.stl_cage_path:
        stl_paths["cage"] = Path(ss.stl_cage_path)

    set_status("running", "배치 해석 초기화 중...")
    ss.progress  = 0
    ss.log_lines = []
    add_log(f"배치 해석 시작: {len(speeds)*len(angles)}개 케이스")

    manager = BatchAnalysisManager(
        mode=mode,
        stl_paths=stl_paths,
        speeds=speeds,
        angles=angles,
        output_csv=csv_path,
        common_params={"n_cores": n_cores},
        progress_cb=lambda p, s, e: (
            setattr(ss, "progress", p),
            setattr(ss, "current_step", f"케이스 {s}/{e}")
        ),
        log_cb=add_log
    )
    ss.batch_manager = manager

    def _run():
        try:
            manager.run_batch()
            set_status("done", "배치 해석 완료!")
            add_log(f"✅ 배치 완료! CSV: {csv_path}")
        except Exception as e:
            set_status("error", str(e))
            add_log(f"❌ 배치 오류: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    ss.job_thread = thread
    thread.start()
    st.rerun()


def _stop_analysis():
    """해석 중지"""
    if ss.job_runner:
        ss.job_runner.stop()
    if ss.batch_manager:
        ss.batch_manager.stop()
    set_status("idle", "사용자에 의해 중지됨")
    add_log("⏹️ 해석 중지됨")
    st.rerun()
