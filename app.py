"""
app.py  ─  양식 가두리 CFD 해석 시스템 메인 대시보드
=============================================================
OpenFOAM 기반 수산공학 CFD 자동화 시스템
Streamlit + PyVista 통합 GUI

실행: streamlit run app.py --server.port 8501
"""

import os
import re
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
from typing import Optional, Dict, List, Tuple

import streamlit as st
from streamlit import session_state as ss

# 백그라운드 스레드에서 Streamlit session_state 접근을 가능하게 하는 컨텍스트 API.
# Streamlit 버전에 따라 경로가 다를 수 있어 예외 처리로 안전하게 import.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # pragma: no cover - 구버전 호환
    try:
        from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx
    except Exception:
        add_script_run_ctx = None
        get_script_run_ctx = None

# ─── 경로 설정 ────────────────────────────────────────────────────────────
APP_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "modules"))

from cfd_manager import (
    UnitCellCaseBuilder, FullStructureCaseBuilder,
    OpenFOAMRunner, ResultExtractor, BatchAnalysisManager,
    get_cpu_count, RESULTS_DIR, STL_UPLOAD_DIR, LOGS_DIR, BASE_DIR,
    TRANSIENT_TURBULENCE_MODELS, compute_transient_stats, read_force_history,
    twine_resolution, unit_cell_base_mm, TWINE_CELLS_TARGET,
    validate_unit_cell_stl, critical_dimension, mesh_adequacy,
    mesh_adequacy_table, SURF_CELLS_TARGET, WAKE_CELLS_TARGET,
    net_grid_base_cell, classify_stl, STL_TYPE_LABELS, STL_TYPE_PRESETS,
    FLUID_PRESETS, estimate_mesh_size, unit_cell_base_mm as _uc_base_mm,
    compute_surface_area, unit_cell_base_mm,
    result_reliability, preflight_checks, steady_force_convergence,
    run_mesh_independence, mesh_independence_table, REFERENCE_CASES,
    write_cylinder_stl, reference_comparison, count_mesh_cells,
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
        "progress":           0.0,
        "current_step":       "",
        "log_lines":          [],
        "last_result_dir":    None,
        "auto_viz":           None,
        "viz_images":         {},
        "results_csv":        None,
        "refresh_interval":   3,
        "job_start_time":     None,
        "refresh_count":      0,
        "case_progress":      {},
        "n_cores":            get_cpu_count(),
        "stl_net_path":       None,
        "stl_cage_path":      None,
        "unit_nx":            1,
        "unit_ny":            1,
        "cell_size_mm":       20.0,
        "solidity_input":     0.15,
        "auto_cell_size_mm":  None,
        "auto_solidity":      None,
        "auto_frontal_area":  None,
        "auto_wire_d_mm":     None,
        # ── 기준면적 Aref: "자동" = 형상에서 계산, "직접 입력" = 사용자 지정값 사용 ──
        "aref_mode":          "자동",
        "aref_manual_m2":     0.0,
        "show_surface_area":  False,   # STL 미리보기 표면적 영역 녹색 표시 토글
        # ── Solver 선택 (기본값은 반드시 Steady = 기존 동작 100% 유지) ──
        # 형상의 가장 가는 치수 기준으로 정밀화 레벨을 자동 상향(STL 종류 무관)
        "auto_refine":        True,
        # ── 전체구조 격자 설계(그물처럼 가는 요소가 흩어진 형상용) ──
        # 기본값은 종전 동작과 완전히 동일하다(재설계 꺼짐 · 후류 자동).
        "net_grid_redesign":  False,
        "net_grid_target_cells": 75.0,
        "wake_box_mode":      "자동",
        "wake_box_level":     2,
        # ── Phase 2: 유체 프리셋 · STL 유형 · Reynolds 대표 길이 ──
        "fluid_type":         "해수 (20℃)",
        "_fluid_applied":     "해수 (20℃)",   # 프리셋 재적용 방지(기본값과 동일)
        "stl_type_user":      "자동 판별 결과 사용",
        "re_length_mode":     "자동",
        "re_length_manual_mm": 0.0,
        # ── Phase 3: 검증 도구 상태 ──
        "mi_levels":          [3, 4, 5],
        "mi_rows":            None,
        "mi_done":            False,
        "ref_case_key":       "sphere",
        "ref_speed":          1.0,
        "ref_result":         None,
        "ref_done":           False,
        # 임계 최소 치수(격자가 반드시 해상해야 할 치수) — 자동/수동
        "crit_dim_mode":      "자동",
        "crit_dim_manual_mm": 0.0,
        "solver_mode":        "Steady (simpleFoam)",
        "tr_end_time":        30.0,    # 물리시간 [s]
        "tr_delta_t":         0.001,
        "tr_max_co":          0.8,     # 보완⑤: 원본 1.0 → 0.8
        "tr_max_delta_t":     0.01,
        "tr_write_interval":  0.5,
        "tr_n_outer":         1,
        "tr_n_corr":          2,
        "tr_n_non_orth":      0,
        "tr_turbulence":      "kOmegaSST",
        "tr_init_steady":     True,    # simpleFoam 선행 수렴을 초기조건으로
        "tr_steady_iters":    1000,
        "tr_perturb":         False,   # 보완③: 대칭 교란
        "tr_perturb_mag":     0.01,
        "tr_avg_start":       0.0,     # 0 이면 자동(구간의 50%)
        # ── 계산량 프리셋 ──
        "calc_preset_name":   "보통",
        "end_time_preset":    2000,
        "refine_level_preset": 3,
        "residual_preset":    "1e-4",
        "write_interval_preset": 100,
        # ── 파이프라인 단계별 진행 상태 ──
        "pipeline_steps":     [],
        # ── 새로고침 복구 표시 플래그 ──
        "_detached_view":     False,
        "_input_restored":    False,
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

init_session()

# ─── 권장 설정(프리셋) 지연 적용 ─────────────────────────────────────────
# Streamlit 은 위젯이 만들어진 뒤 그 key 의 session_state 를 바꾸는 것을 막는다
# (StreamlitAPIException). 그래서 프리셋 적용 버튼은 값을 바로 쓰지 않고
# _pending_preset 에 담아 rerun 하고, 위젯이 만들어지기 전인 여기서 반영한다.
_pending = ss.pop("_pending_preset", None)
if _pending:
    for _k, _v in _pending.items():
        ss[_k] = _v
    ss["_preset_applied_msg"] = _pending.get("_msg", "권장 설정을 적용했습니다.")


# ═══════════════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════

def add_log(msg: str):
    ss.log_lines.append(f"[{datetime.now():%H:%M:%S}] {msg}")
    if len(ss.log_lines) > 500:
        ss.log_lines = ss.log_lines[-400:]
    _persist_job_state()          # 새로고침 복구용 디스크 기록 (스로틀됨)

def set_status(status: str, step: str = ""):
    ss.job_status   = status
    ss.current_step = step
    _persist_job_state(force=True)   # 상태 전환은 즉시 기록


# ─── 작업 상태 영속화 (새로고침/세션 단절 후 진행률·로그 복구) ──────────────
JOB_STATE_FILE = RESULTS_DIR / ".job_state.json"
_last_persist  = {"ts": 0.0}

def _persist_job_state(force: bool = False):
    """실행 중인 작업 상태를 디스크에 기록한다. 백그라운드 스레드가 호출하므로,
    브라우저를 새로고침해 세션이 바뀌어도 새 세션이 이 파일을 읽어 복구할 수 있다.
    과도한 쓰기를 막기 위해 force가 아니면 1.5초 간격으로 스로틀한다."""
    now = time.time()
    if not force and (now - _last_persist["ts"] < 1.5):
        return
    _last_persist["ts"] = now
    try:
        state = {
            "job_status":      ss.get("job_status", "idle"),
            "progress":        float(ss.get("progress", 0.0)),
            "current_step":    ss.get("current_step", ""),
            "last_result_dir": ss.get("last_result_dir"),
            "job_start_time":  ss.get("job_start_time"),
            "case_progress":   ss.get("case_progress", {}),
            "pipeline_steps":  ss.get("pipeline_steps", []),
            "log_tail":        (ss.get("log_lines", []) or [])[-200:],
            "updated_at":      now,
        }
        tmp = JOB_STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False))
        tmp.replace(JOB_STATE_FILE)        # 원자적 교체 (부분 읽기 방지)
    except Exception:
        pass

def _load_job_state() -> Optional[dict]:
    try:
        if JOB_STATE_FILE.exists():
            return json.loads(JOB_STATE_FILE.read_text())
    except Exception:
        return None
    return None

def _restore_job_state_if_detached():
    """이 세션이 실행 중인 작업을 직접 소유(job_thread 보유)하지 않는데
    디스크 상태가 '실행 중'이면, 진행률·로그·파이프라인을 복구해 화면에 표시한다.
    180초 이상 갱신이 없으면 백그라운드 프로세스가 죽은 것으로 보고 오류 표시."""
    if ss.get("job_thread") is not None:
        ss._detached_view = False
        return  # 작업 소유 세션 → 라이브 ss 사용
    data = _load_job_state()
    if not data:
        ss._detached_view = False
        return
    status       = data.get("job_status")
    age          = time.time() - data.get("updated_at", 0)
    was_detached = ss.get("_detached_view", False)

    if status == "running" and age > 1800:
        # 30분 이상 방치된 상태 파일 → 오래전에 죽은 작업으로 보고 무시
        ss._detached_view = False
        return

    if status == "running":
        # 실행 중인 작업 복구 (진행률·로그·파이프라인)
        ss.progress        = float(data.get("progress", 0.0))
        ss.current_step    = data.get("current_step", "")
        ss.last_result_dir = data.get("last_result_dir")
        ss.job_start_time  = data.get("job_start_time")
        ss.case_progress   = data.get("case_progress", {})
        ss.pipeline_steps  = data.get("pipeline_steps", [])
        ss.log_lines       = data.get("log_tail", [])
        ss._detached_view  = True
        if age > 180:
            # 180초 이상 갱신 없음 → 백그라운드 프로세스 종료/중단으로 간주
            ss.job_status   = "error"
            ss.current_step = "백그라운드 작업이 응답하지 않습니다 (서버 재시작·종료 추정)"
        else:
            ss.job_status   = "running"
    elif status in ("done", "error") and was_detached:
        # 새로고침으로 지켜보던 작업이 끝남 → 최종 상태를 한 번 반영
        ss.progress       = float(data.get("progress", 0.0))
        ss.current_step   = data.get("current_step", "")
        ss.pipeline_steps = data.get("pipeline_steps", [])
        ss.log_lines      = data.get("log_tail", [])
        ss.job_status     = status
        ss._detached_view = False
    else:
        ss._detached_view = False

# ─── 입력(STL 업로드) 상태 영속화 (새로고침 후 업로드·미리보기 복구) ────────
INPUT_STATE_FILE = RESULTS_DIR / ".input_state.json"

# 새로고침·새 세션에서 복구할 입력 항목.
# [결함 수정] 종전에는 STL·셀 크기·기준면적 등 일부만 저장해, 새로고침하면
# 유속·영각 범위와 계산량 프리셋(반복·정밀화·수렴·저장간격), 물리 조건(rho·nu·
# 난류강도), Solver 설정이 조용히 기본값으로 돌아갔다. 긴 해석 도중 진행 상황을
# 보려고 새로고침하면 입력이 초기화되는 문제였다.
# (실측: F5 후 최대유속 2.0→1.0, 유속단계 3→1, 최대영각 45→0, nu 1.05→1.19,
#  최대반복 5000→2000, 저장간격 200→100 으로 복귀)
_INPUT_STATE_KEYS = (
    # 형상·감지 (종전부터 저장하던 항목)
    "analysis_mode", "stl_net_path", "stl_cage_path",
    "auto_cell_size_mm", "auto_wire_d_mm", "auto_solidity",
    "auto_frontal_area", "cell_size_mm", "solidity_input",
    "unit_nx", "unit_ny", "aref_mode", "aref_manual_m2", "active_project",
    # 해석 조건
    "u_min", "u_max", "u_steps", "a_min", "a_max", "a_steps",
    # 물리 조건
    "rho", "nu", "ti", "fluid_type", "_fluid_applied",
    # 계산량 프리셋
    "calc_preset_name", "end_time_preset", "refine_level_preset",
    "residual_preset", "write_interval_preset",
    # 격자 옵션
    "auto_refine", "crit_dim_mode", "crit_dim_manual_mm",
    "net_grid_redesign", "net_grid_target_cells",
    "wake_box_mode", "wake_box_level",
    # 전체구조 치수
    "cage_d", "cage_h",
    # Solver(비정상 포함)
    "solver_mode", "tr_end_time", "tr_delta_t", "tr_max_co", "tr_max_delta_t",
    "tr_write_interval", "tr_n_outer", "tr_n_corr", "tr_n_non_orth",
    "tr_turbulence", "tr_init_steady", "tr_steady_iters", "tr_perturb",
    "tr_perturb_mag", "tr_avg_start",
    # 판별·기준 길이
    "stl_type_user", "re_length_mode", "re_length_manual_mm",
)


def _persist_input_state():
    """업로드한 STL 선택과 자동 감지 정보를 디스크에 기록한다. STL 파일 자체는
    이미 stl_uploads/ 에 저장돼 있으므로, 여기서는 '어떤 파일을 쓰는지'와 감지
    결과만 저장해 두면 새로고침 후 새 세션이 그대로 복구할 수 있다."""
    # v13 항목1: 활성 프로젝트가 있으면 STL 이 없어도 상태를 기록한다(F5 후 복구).
    if not (ss.get("stl_net_path") or ss.get("stl_cage_path")
            or ss.get("active_project")):
        return
    try:
        state = {k: ss.get(k) for k in _INPUT_STATE_KEYS}
        tmp = INPUT_STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, default=str))
        tmp.replace(INPUT_STATE_FILE)
    except Exception:
        pass

def _restore_input_state():
    """새 세션(F5 등)에서 이전 STL 선택·감지 정보를 복구한다.
    저장된 STL 파일이 디스크에 실제 존재할 때만 복구한다."""
    if ss.get("stl_net_path") or ss.get("stl_cage_path"):
        return  # 이미 이 세션에 STL이 있으면 그대로 둠
    try:
        if not INPUT_STATE_FILE.exists():
            return
        data = json.loads(INPUT_STATE_FILE.read_text())
    except Exception:
        return
    _net  = data.get("stl_net_path")
    _cage = data.get("stl_cage_path")
    _has_net  = bool(_net)  and Path(_net).exists()
    _has_cage = bool(_cage) and Path(_cage).exists()
    if not (_has_net or _has_cage):
        return  # 파일이 사라졌으면 복구하지 않음
    if _has_net:
        ss.stl_net_path = _net
    if _has_cage:
        ss.stl_cage_path = _cage
    # [순서 주의] 활성 프로젝트 로드가 먼저다.
    # v13 항목1: 활성 프로젝트가 있었으면 새로고침 후 자동으로 다시 로드해
    # 조건·결과·매트릭스를 복원한다(프로젝트 폴더가 실제 존재할 때만).
    # _restore_input_state 는 _pending_load_project 처리 지점보다 뒤에서
    # 호출되므로 즉시 반영을 위해 직접 로드한다.
    _ap = data.get("active_project")
    if _ap:
        _apmode = data.get("analysis_mode") or ss.get("analysis_mode", "unit_cell")
        if (_project_dir(_apmode, _ap) / "project.json").exists():
            _apply_project_load(_apmode, _ap)
            ss["_last_synced_proj_name"] = None

    # 그 다음에 '마지막 화면 값'을 덮어쓴다.
    # [결함 수정] 종전에는 순서가 반대라, 프로젝트 저장 시점 이후에 사용자가
    # 바꾼 값(미저장 편집)이 새로고침 때 프로젝트 값으로 되돌아갔다. 실측:
    # 최대유속 2.0→1.0, 유속단계 3→1, 최대영각 45→0, nu 1.05→1.19,
    # 최대반복 5000→2000, 저장간격 200→100. 미저장 편집은 그대로 두고,
    # 저장 여부는 기존 '미저장 변경' 가드가 계속 알려준다.
    for k in _INPUT_STATE_KEYS:
        if k in ("stl_net_path", "stl_cage_path", "active_project"):
            continue
        if data.get(k) is not None:
            ss[k] = data[k]
    ss._input_restored = True

# ─── 프로젝트(케이스 묶음) 저장/불러오기/새로 만들기 (항목3·4) ────────────────
# 프로젝트 = 사용자가 만든 '해석 조건 + 해석 결과(결과 CSV) + 물리/계산/시각화 설정'의
# 묶음. 결과 CSV 파일명이 프로젝트의 결과 데이터 식별자가 된다(항목5: CSV 탭은 활성
# 프로젝트 데이터만 표시). 메타데이터(이름·모드·저장시각)를 함께 보관하며, 키 목록만
# 늘리면 항목을 확장할 수 있다(확장성).
PROJECT_KEYS = [
    # 해석 조건(유속·영각 범위)
    "u_min", "u_max", "u_steps", "a_min", "a_max", "a_steps",
    # 물리 조건
    "rho", "nu", "ti",
    # 계산량 프리셋
    "end_time_preset", "refine_level_preset", "residual_preset",
    "write_interval_preset",
    # 형상/감지/타일
    "analysis_mode", "stl_net_path", "stl_cage_path",
    "auto_cell_size_mm", "auto_wire_d_mm", "auto_solidity",
    "auto_frontal_area", "cell_size_mm", "solidity_input",
    "unit_nx", "unit_ny",
    # 기준면적 Aref(자동/직접 입력)
    "aref_mode", "aref_manual_m2",
    # 격자 자동 보정 / Solver 선택 및 비정상 해석 설정
    "auto_refine", "crit_dim_mode", "crit_dim_manual_mm",
    "net_grid_redesign", "net_grid_target_cells", "wake_box_mode", "wake_box_level",
    # Phase 2: 유체 프리셋·STL 유형·대표 길이
    "fluid_type", "stl_type_user", "re_length_mode", "re_length_manual_mm",
    "solver_mode", "tr_end_time", "tr_delta_t", "tr_max_co", "tr_max_delta_t",
    "tr_write_interval", "tr_n_outer", "tr_n_corr", "tr_n_non_orth",
    "tr_turbulence", "tr_init_steady", "tr_steady_iters", "tr_perturb",
    "tr_perturb_mag", "tr_avg_start",
    # 결과 CSV(프로젝트 데이터) + 시각화 설정
    "batch_csv_name",
    "r1_viewmode", "r1_field", "r1_opacity_vol", "r1_opacity_iso",
]
# '새 프로젝트' 시 초기화할 기본값(깨끗한 상태)
PROJECT_DEFAULTS = {
    "u_min": 1.0, "u_max": 1.0, "u_steps": 1,
    "a_min": 0.0, "a_max": 0.0, "a_steps": 1,
    "rho": 1025.0, "nu": 1.19, "ti": 5,
    "aref_mode": "자동", "aref_manual_m2": 0.0,
    "r1_viewmode": "슬라이스", "r1_field": "U",
    "r1_opacity_vol": 0.55, "r1_opacity_iso": 0.55,
}

def _projects_dir(_mode):
    return RESULTS_DIR / _mode / "projects"

def _project_dir(_mode, _name):
    return _projects_dir(_mode) / _name

def _active_project_dir():
    """활성 프로젝트 폴더 경로(없으면 None)."""
    _name = ss.get("active_project")
    if not _name:
        return None
    return _project_dir(ss.get("analysis_mode", "unit_cell"), _name)

def _results_root(_mode):
    """결과(케이스 디렉토리·CSV)를 저장/스캔하는 루트.
    활성 프로젝트가 있으면 그 폴더(항목1: 모든 산출물을 프로젝트 폴더 안에),
    없으면 모드 루트(레거시/임시 작업)."""
    _pd = _active_project_dir()
    return _pd if _pd is not None else (RESULTS_DIR / _mode)

def _project_csv_path(_mode, _csv_name):
    """결과 CSV 경로 = 결과 루트 / 파일명."""
    return _results_root(_mode) / _csv_name

def list_project_names(_mode):
    _d = _projects_dir(_mode)
    if not _d.exists():
        return []
    return sorted(p.name for p in _d.iterdir()
                  if p.is_dir() and (p / "project.json").exists())

def save_project(_mode, _name):
    """현재 세션 상태를 프로젝트 폴더의 project.json 으로 저장."""
    # stl_net_path 등은 Path 객체일 수 있어 JSON 직렬화가 안 된다 → 문자열로 정규화.
    def _ser(v):
        return str(v) if isinstance(v, Path) else v
    _state = {k: _ser(ss.get(k)) for k in PROJECT_KEYS}
    _state["_meta"] = {"name": _name, "mode": _mode,
                       "saved": datetime.now().isoformat()}
    _d = _project_dir(_mode, _name); _d.mkdir(parents=True, exist_ok=True)
    # default=str: 예상치 못한 비직렬화 객체(Path 등)도 안전하게 문자열화.
    (_d / "project.json").write_text(
        json.dumps(_state, ensure_ascii=False, indent=2, default=str))
    ss.active_project = _name

def create_project(_mode, _name):
    """새 프로젝트 폴더 생성 + 깨끗한 상태로 초기화(항목1·2).
    이후 모든 설정·조건·중간파일·결과는 이 폴더 안에 저장된다."""
    for k, v in PROJECT_DEFAULTS.items():
        ss[k] = v
    ss["batch_csv_name"] = "force_coeffs.csv"   # 프로젝트 폴더 내부 파일
    ss["res_sel_cond"] = None
    ss["result_csv_sel"] = None
    ss.active_project = _name
    save_project(_mode, _name)                  # 폴더 + project.json 생성

def _apply_project_load(_mode, _name):
    """프로젝트 project.json 을 세션 상태에 반영(위젯 생성 전 호출돼야 안전)."""
    _p = _project_dir(_mode, _name) / "project.json"
    if not _p.exists():
        return False
    try:
        _data = json.loads(_p.read_text())
    except Exception:
        return False
    for k in PROJECT_KEYS:
        if k in _data and _data[k] is not None:
            ss[k] = _data[k]
    ss.active_project = _name
    ss.res_sel_cond = None          # 결과 선택 초기화(불러온 프로젝트 데이터로)
    return True

def _apply_project_new():
    """프로젝트 미선택(임시 작업) 상태로 초기화 — 다이얼로그 취소/호환용."""
    for k, v in PROJECT_DEFAULTS.items():
        ss[k] = v
    for k in ("stl_net_path", "stl_cage_path", "batch_csv_name",
              "res_sel_cond", "result_csv_sel"):
        ss[k] = None
    ss.active_project = None


def _project_changed_keys(_mode):
    """저장된 project.json 과 다른 PROJECT_KEYS 를 [(키, 현재값, 저장값)] 로 반환.
    가드 다이얼로그에 '무엇이 바뀌었는지' 보여주고, 오탐을 진단하는 데 쓴다."""
    _name = ss.get("active_project")
    if not _name:
        return [(k, ss.get(k), v) for k, v in PROJECT_DEFAULTS.items()
                if k in ss and ss.get(k) != v]
    _p = _project_dir(_mode, _name) / "project.json"
    if not _p.exists():
        return [("(저장본 없음)", _name, None)]
    try:
        _saved = json.loads(_p.read_text())
    except Exception:
        return [("(저장본 손상)", _name, None)]

    def _norm(v):
        # Path→str, 그리고 1 과 1.0 처럼 JSON 왕복으로 타입만 달라진 수치는
        # 같은 값으로 본다(오탐 방지 — 이것 때문에 저장 직후에도 '미저장 변경'
        # 으로 잡혀 불러오기마다 가드가 떴다).
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v)
        return v

    return [(k, ss.get(k), _saved.get(k)) for k in PROJECT_KEYS
            if _norm(ss.get(k)) != _norm(_saved.get(k))]


def _project_has_unsaved_changes(_mode):
    """현재 세션의 프로젝트 설정이 저장된 project.json 과 다른지(항목5).
    활성 프로젝트가 없으면(임시 작업) 변경 여부를 판단할 기준이 없으므로,
    조건/결과가 하나라도 설정돼 있으면 '미저장'으로 본다."""
    if not ss.get("active_project"):
        return bool(_project_changed_keys(_mode)
                    or ss.get("stl_net_path") or ss.get("stl_cage_path"))
    return bool(_project_changed_keys(_mode))


@st.dialog("💾 변경사항을 저장할까요?")
def _unsaved_guard_dialog(_next_action: str):
    """항목5: 현재 프로젝트에 미저장 변경이 있을 때, 파괴적 동작(새 프로젝트·
    다른 프로젝트 불러오기) 직전에 저장 여부를 확인한다."""
    _mode = ss.get("analysis_mode", "unit_cell")
    _act_label = {"new": "새 프로젝트 만들기",
                  "load": "다른 프로젝트 불러오기"}.get(_next_action, "계속")
    st.write(f"현재 프로젝트에 저장하지 않은 변경사항이 있습니다. "
             f"**{_act_label}** 전에 저장할까요?")
    _chg = _project_changed_keys(_mode)
    if _chg:
        with st.expander(f"변경된 항목 {len(_chg)}개 보기", expanded=False):
            for _k, _cur, _sav in _chg[:20]:
                st.caption(f"· **{_k}**:  저장값 `{_sav}`  →  현재 `{_cur}`")
    _c1, _c2, _c3 = st.columns(3)
    if _c1.button("💾 저장하고 계속", type="primary", use_container_width=True):
        _nm = (ss.get("active_project")
               or (ss.get("proj_name_input") or "").strip())
        if _nm:
            save_project(_mode, _nm)
        ss["_guard_proceed"] = _next_action
        st.rerun()
    if _c2.button("저장 안 함", use_container_width=True):
        ss["_guard_proceed"] = _next_action
        st.rerun()
    if _c3.button("취소", use_container_width=True):
        st.rerun()


@st.dialog("🆕 새 프로젝트 만들기")
def _new_project_dialog():
    """항목1: 프로젝트 이름을 입력받아 폴더를 생성한다. 동일 이름이 이미 있으면
    덮어쓰기 확인을 묻고, '예'면 기존 프로젝트를 덮어쓰고 진행, '아니오/취소'면
    생성을 취소(또는 다른 이름 입력)한다."""
    st.write("새 프로젝트 이름을 입력하세요. 입력한 이름으로 폴더가 생성되고, "
             "이후 모든 설정·조건·중간파일·결과가 그 폴더에 저장됩니다.")
    _nm = (st.text_input("프로젝트 이름", key="new_proj_name_input",
                         placeholder="예: onemesh2_실험1") or "").strip()
    _mode = ss.get("analysis_mode", "unit_cell")
    _exists = _nm in list_project_names(_mode) if _nm else False
    _c1, _c2 = st.columns(2)
    if _exists:
        st.warning("⚠️ 동일 이름의 프로젝트가 이미 있습니다. 덮어쓸까요?")
        if _c1.button("예, 덮어쓰기", type="primary", use_container_width=True):
            ss._pending_overwrite_project = _nm
            st.rerun()
        if _c2.button("아니오 / 취소", use_container_width=True):
            # 생성 취소(다이얼로그 닫기). 다른 이름을 쓰려면 이름 칸을 수정하면
            # 경고가 사라지고 '만들기'가 활성화된다.
            st.rerun()
    else:
        if _c1.button("만들기", type="primary", use_container_width=True,
                      disabled=(not _nm)):
            ss._pending_create_project = _nm
            st.rerun()
        if _c2.button("취소", use_container_width=True):
            st.rerun()

# v13 항목5: 미저장 변경 가드 다이얼로그 결과 처리(위젯 생성 전).
_gp = ss.pop("_guard_proceed", None)
if _gp == "new":
    _new_project_dialog()
elif _gp == "load":
    _tgt = ss.pop("_pending_load_after_guard", None) or ss.get("proj_load_sel")
    if _tgt:
        ss._pending_load_project = _tgt
        ss["_last_synced_proj_name"] = None
        ss["_last_synced_load_sel"] = None

# 위젯 생성 '이전'에 보류된 로드/생성/덮어쓰기/새프로젝트를 적용(세션 상태 안전 변경).
if ss.get("_pending_load_project"):
    _apply_project_load(ss.get("analysis_mode", "unit_cell"),
                        ss.pop("_pending_load_project"))
if ss.get("_pending_create_project"):
    create_project(ss.get("analysis_mode", "unit_cell"),
                   ss.pop("_pending_create_project"))
if ss.get("_pending_overwrite_project"):
    # 항목1: 기존 프로젝트 폴더를 삭제(덮어쓰기) 후 깨끗하게 재생성.
    _ovw_name = ss.pop("_pending_overwrite_project")
    _ovw_mode = ss.get("analysis_mode", "unit_cell")
    try:
        _ovw_dir = _project_dir(_ovw_mode, _ovw_name)
        if _ovw_dir.exists():
            shutil.rmtree(_ovw_dir, ignore_errors=True)
    except Exception:
        pass
    create_project(_ovw_mode, _ovw_name)
if ss.pop("_pending_new_project", False):
    _apply_project_new()

# 매 렌더링마다 디스크 상태 확인 → 새로고침 후 진행 중인 작업 자동 복구
_restore_job_state_if_detached()
# 새로고침 후 STL 업로드 선택·미리보기 복구
_restore_input_state()

def status_badge(status: str) -> str:
    labels = {"idle":"대기","running":"해석 중","done":"완료","error":"오류"}
    return f'<span class="badge-{status}">{labels.get(status, status)}</span>'

# ─── 예상 소요 시간 추정 ────────────────────────────────────────────────────
def estimate_case_minutes(end_time: int, refine_level: int, n_cores: int,
                          mode: str = "unit_cell") -> float:
    """단일 케이스 예상 소요(분) — 전 워크플로 기여분 합산(항목1):
        전처리(surfaceFeatureExtract·decomposePar) + 메싱(직렬 snappy) +
        솔버(병렬 simpleFoam) + 후처리(reconstructPar·I/O·시각화 준비).
    정밀화 레벨이 격자 셀 수를, 솔버는 반복수에 비례·코어수에 반비례한다고 본다.

    모드별 격자 규모 차이를 반영: unit_cell 은 주기 1셀(소형), full_structure 는
    개방 도메인(대형 격자) → 메싱·솔버·후처리가 모두 더 크다. 경험식은 '초기(cold-start)
    추정'이며, 케이스 완료 시 실측이 모드별로 누적되면 estimate_total_minutes 가 실측
    중앙값으로 자동 보정해 예측-실측 오차를 줄인다.

    재보정(2026-06-22, unit_cell): 최소(500·lvl2·16코어) 실측 ≈1.2분.
    """
    f_iter   = max(1, end_time) / 2000.0
    f_refine = {1: 0.3, 2: 0.6, 3: 1.0, 4: 2.2, 5: 5.0}.get(int(refine_level), 1.0)
    f_cores  = 16.0 / max(1, int(n_cores))
    if mode == "full_structure":
        # 개방 도메인 대형 격자: 전처리·직렬 메싱·솔버·후처리 모두 가중.
        pre_min    = 0.4 + 0.3 * f_refine                 # sfe + decomposePar(대형)
        mesh_min   = 0.8 + 1.4 * f_refine                 # 직렬 snappy(대형 격자)
        solver_min = 8.0 * f_iter * f_refine * f_cores    # 대형 격자 병렬 솔버
        post_min   = 0.5 + 0.5 * f_refine                 # reconstructPar(16proc 재조합)+I/O
    else:  # unit_cell (소형 주기 격자)
        pre_min    = 0.1
        mesh_min   = 0.4 + 0.6 * f_refine                 # 직렬 snappy 메싱
        solver_min = 4.0 * f_iter * f_refine * f_cores    # 병렬 솔버
        post_min   = 0.2                                  # 경량 후처리/IO
    return pre_min + mesh_min + solver_min + post_min


# ─── 실측 기반 추정 보정(항목1) ─────────────────────────────────────────────
# 완료된 케이스의 실제 소요(분)를 모드별로 누적 기록하고, 그 중앙값을 다음 배치의
# 케이스당 추정에 사용한다. 경험식만으로는 환경·격자 변화에 따라 빗나가므로,
# 실제 실행시간이 쌓일수록 추정이 실측에 수렴한다.
def _timing_store_path(mode):
    return RESULTS_DIR / mode / ".case_timing.json"

def record_case_minutes(mode, minutes, end_time=None, refine_level=None,
                        n_cores=None):
    """완료 케이스의 실제 소요(분)와 그때의 설정(반복·정밀도·코어)을 함께 기록한다
    (최근 30개). 설정을 저장해야 다음 추정에서 '현재 설정으로 스케일링'할 수 있다."""
    try:
        # 항목2: 0.1분(6초) 미만은 메싱·솔버를 실제로 수행한 케이스일 수 없다(빠른
        # 실패·중단 등 가비지). 보정 추정을 오염시키므로 기록하지 않는다.
        if not (minutes and float(minutes) >= 0.1):
            return
        p = _timing_store_path(mode)
        hist = []
        if p.exists():
            try:
                hist = json.loads(p.read_text())
            except Exception:
                hist = []
        rec = {"min": round(float(minutes), 3)}
        if end_time is not None:     rec["et"] = int(end_time)
        if refine_level is not None: rec["rl"] = int(refine_level)
        if n_cores is not None:      rec["nc"] = int(n_cores)
        hist.append(rec)
        # dict 레코드만 유지(구버전 float 기록은 폐기)
        hist = [r for r in hist if isinstance(r, dict)][-30:]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(hist))
    except Exception:
        pass

def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else None

def estimate_total_minutes(mode, n_cases, end_time, refine_level, n_cores):
    """배치 총 예상(분). 항목3: 최근 실측 각각을 '경험식 비'(현재 설정/측정 당시
    설정)로 스케일링한 뒤 중앙값을 사용 → 반복·정밀도·코어가 바뀌어도 실측에
    기반해 정확히 추정한다. 기록이 없으면 경험식으로 폴백."""
    per = None
    try:
        p = _timing_store_path(mode)
        if p.exists():
            # 항목2: 비현실적으로 작은(≥0.1분 미만) 가비지 기록은 추정에서 제외 →
            # '총 < 1분' 같은 모순 방지. 유효 기록이 없으면 경험식으로 폴백.
            recs = [r for r in json.loads(p.read_text())
                    if isinstance(r, dict) and float(r.get("min", 0)) >= 0.1]
            if recs:
                _cur = estimate_case_minutes(end_time, refine_level, n_cores, mode)
                _scaled = []
                for r in recs:
                    _base = estimate_case_minutes(
                        r.get("et", end_time), r.get("rl", refine_level),
                        r.get("nc", n_cores), mode)
                    _scaled.append(float(r["min"]) * (_cur / _base if _base > 0 else 1.0))
                per = _median(_scaled)
    except Exception:
        per = None
    if per is None:
        per = estimate_case_minutes(end_time, refine_level, n_cores, mode)
    return per * max(1, int(n_cases))

def fmt_duration(minutes: float) -> str:
    """분 단위 시간을 '시간 분' 형식 문자열로."""
    if minutes < 1:
        return "1분 미만"
    m = int(round(minutes))
    if m < 60:
        return f"{m}분"
    h, mm = divmod(m, 60)
    return f"{h}시간 {mm}분" if mm else f"{h}시간"

def fmt_elapsed(seconds: float) -> str:
    """초 단위 경과 시간을 항상 '시 분 초' 형식으로 (예: 0시간 05분 03초,
    1시간 23분 45초). 사용자 요청에 따라 1시간 미만에도 '시간'을 명시한다."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}시간 {m:02d}분 {sec:02d}초"

def save_uploaded_stl(uploaded_file, prefix: str) -> Optional[Path]:
    """업로드된 STL 파일을 임시 디렉토리에 저장"""
    if uploaded_file is None:
        return None
    save_path = STL_UPLOAD_DIR / f"{prefix}_{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def render_stl_preview(stl_path: str, angle_deg: float = 0.0) -> Optional[bytes]:
    """STL 형상을 PyVista로 렌더링하여 PNG bytes 반환. 실패 시 matplotlib 폴백."""
    import io
    try:
        import pyvista as pv
        pv.global_theme.font.family = "courier"
        mesh = pv.read(str(stl_path))
        mesh_mm = mesh.scale(1000.0, inplace=False)  # m→mm 표시용
        bounds = mesh_mm.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)

        pl = pv.Plotter(off_screen=True, window_size=[700, 520])
        pl.set_background("#0d1117")
        pl.add_mesh(mesh_mm, color="#5dade2", opacity=0.85, show_edges=True,
                    edge_color="#a9cce3", line_width=0.5)

        # 좌표축
        pl.add_axes(color="white", xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]")

        # 유속 방향 화살표 (빨간색)
        _theta = math.radians(angle_deg)
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        span = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        arrow_len = span * 0.7
        ox = cx - math.cos(_theta) * arrow_len
        oz = cz - math.sin(_theta) * arrow_len
        arrow = pv.Arrow(
            start=(ox, cy, oz),
            direction=(math.cos(_theta), 0, math.sin(_theta)),
            scale=arrow_len, tip_length=0.25, tip_radius=0.08, shaft_radius=0.03
        )
        pl.add_mesh(arrow, color="#e74c3c")
        pl.add_text(f"U (α={angle_deg:.0f}°)", position="upper_left",
                    font_size=10, color="#e74c3c")

        # 바운딩 박스 아웃라인
        pl.add_mesh(mesh_mm.outline(), color="#f39c12", line_width=1.5)
        # 그물면(법선 Z)을 약간 기울여 거의 정면으로 본다 — 얇은 평판이라
        # isometric(모서리) 시점에서는 납작한 조각처럼 보이므로, 법선 쪽으로
        # 기울여 망목(사각 개구부) 패턴이 또렷이 보이게 한다.
        pl.view_vector((0.45, -0.75, 0.95), viewup=(0, 0, 1))
        pl.camera.zoom(1.2)

        buf = io.BytesIO()
        img = pl.screenshot(return_img=True)
        pl.close()
        from PIL import Image
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        pass

    # ── matplotlib 폴백 ──────────────────────────────────────────────────
    try:
        import struct
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        with open(str(stl_path), "rb") as f:
            header = f.read(80)
            n_tri = struct.unpack("<I", f.read(4))[0]
            verts = []
            for _ in range(min(n_tri, 5000)):
                f.read(12)
                pts = [struct.unpack("<fff", f.read(12)) for _ in range(3)]
                f.read(2)
                verts.append(pts)
        verts_mm = [[(x*1000, y*1000, z*1000) for x,y,z in tri] for tri in verts]

        fig = plt.figure(figsize=(7, 5), facecolor="#0d1117")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")
        poly = Poly3DCollection(verts_mm, alpha=0.5, facecolor="#5dade2",
                                edgecolor="#a9cce3", linewidth=0.2)
        ax.add_collection3d(poly)
        xs = [p[0] for tri in verts_mm for p in tri]
        ys = [p[1] for tri in verts_mm for p in tri]
        zs = [p[2] for tri in verts_mm for p in tri]
        ax.set_xlim(min(xs), max(xs)); ax.set_ylim(min(ys), max(ys)); ax.set_zlim(min(zs), max(zs))
        # 실제 치수 비율 유지 (기본값은 정육면체로 왜곡되어 얇은 판이 두껍게 보임)
        _dx, _dy, _dz = max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)
        try:
            ax.set_box_aspect((_dx, _dy, max(_dz, _dx*0.02)))
        except Exception:
            pass
        # 그물면 법선 쪽으로 기울여 본다 (납작한 조각이 아닌 망목 패턴이 보이도록)
        ax.view_init(elev=55, azim=-60)
        ax.set_xlabel("X [mm]", color="white"); ax.set_ylabel("Y [mm]", color="white")
        ax.set_zlabel("Z [mm]", color="white")
        ax.tick_params(colors="white"); ax.title.set_color("white")

        # 유속 화살표
        _theta = math.radians(angle_deg)
        mx, my, mz = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2
        span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) * 0.5
        ax.quiver(mx-math.cos(_theta)*span, my, mz-math.sin(_theta)*span,
                  math.cos(_theta)*span, 0, math.sin(_theta)*span,
                  color="#e74c3c", linewidth=2, arrow_length_ratio=0.3)
        ax.set_title(f"STL 미리보기  |  유속 방향 α={angle_deg:.0f}°", color="white", pad=10)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", facecolor="#0d1117", dpi=110)
        plt.close()
        return buf.getvalue()
    except Exception:
        return None


def render_stl_interactive_plotly(
    stl_path: str,
    mode: str,
    angle_deg: float,
    nx: int = 1,
    ny: int = 1,
    cell_size_m: float = 0.02,   # 폴백용 — 실제 타일 간격은 STL bbox에서 자동 계산
    init_camera: bool = True,
    highlight_surface: bool = False,
    display_frame: bool = True,
):
    """Plotly go.Mesh3d 기반 인터랙티브 3D STL 뷰어. 마우스 드래그로 회전 가능.

    init_camera=False면 scene에서 camera 키를 빼서, 영각·nx·ny 위젯을 바꿔도
    uirevision='stlpreview'가 사용자의 마우스 카메라를 보존하게 한다(첫 렌더만 True).
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        return None

    # ── STL 읽기 (PyVista 우선, binary STL 폴백) ──────────────────────────
    def _read_stl(path):
        """(pts: N×3, faces: M×3 int) 반환.
        STL 좌표는 이미 mm 단위 (blockMeshDict: scale 0.001 적용) — 변환 없음."""
        try:
            import pyvista as pv
            m = pv.read(str(path))
            pts = m.points.copy()              # 좌표 단위: mm (변환 불필요)
            raw = m.faces
            if len(raw) == 0:
                raise ValueError("empty mesh")
            faces = raw.reshape(-1, 4)[:, 1:]  # [3, v0,v1,v2, …] → (M,3)
            return pts.astype(float), faces.astype(int)
        except Exception:
            pass
        # Binary STL 폴백
        import struct
        pts_l, faces_l = [], []
        with open(str(path), "rb") as f:
            f.read(80)
            n = struct.unpack("<I", f.read(4))[0]
            for t in range(n):
                f.read(12)
                base = len(pts_l)
                for _ in range(3):
                    vx, vy, vz = struct.unpack("<fff", f.read(12))
                    pts_l.append([vx, vy, vz])  # mm 단위 그대로 사용
                f.read(2)
                faces_l.append([base, base + 1, base + 2])
        return np.array(pts_l, dtype=float), np.array(faces_l, dtype=int)

    try:
        base_pts, base_faces = _read_stl(stl_path)
    except Exception:
        return None

    # ── 타일 간격: STL 바운딩박스에서 직접 계산 (단위 의존성 없음) ────────
    bx_min = float(base_pts[:, 0].min()); bx_max = float(base_pts[:, 0].max())
    by_min = float(base_pts[:, 1].min()); by_max = float(base_pts[:, 1].max())
    bz_min = float(base_pts[:, 2].min()); bz_max = float(base_pts[:, 2].max())

    step_x = (bx_max - bx_min) or (cell_size_m * 1000.0)
    step_y = (by_max - by_min) or (cell_size_m * 1000.0)

    # ── 단위셀 모드: Nx × Ny 타일링 ──────────────────────────────────────
    do_tile = (mode == "unit_cell") and (nx > 1 or ny > 1)
    offsets = [(ix, iy) for ix in range(nx) for iy in range(ny)] if do_tile else [(0, 0)]

    n_base = len(base_pts)
    all_pts, all_faces = [], []
    for step_idx, (ix, iy) in enumerate(offsets):
        shift = np.array([ix * step_x, iy * step_y, 0.0])
        all_pts.append(base_pts + shift)
        all_faces.append(base_faces + step_idx * n_base)

    pts   = np.vstack(all_pts)
    faces = np.vstack(all_faces)

    # ── 통일 표시 좌표계 변환 (유동장 뷰와 축을 일치시킴) ──────────────────
    # 유동장은 solver_to_display_rotation(mode, α) 로 회전해 그려진다. 미리보기가
    # 원좌표를 쓰면 같은 형상이 두 화면에서 다른 축으로 보이므로 동일 변환을 건다.
    #
    #   unit_cell      : R = [[0,0,1],[-1,0,0],[0,-1,0]]  (α 무관)
    #   full_structure : 케이스가 형상을 Y축 α 회전하므로 실제 표시 변환은
    #                    R(α)·M_y(α) 인데, 이 곱은 α 와 무관하게 위와 같은
    #                    상수 행렬이 된다(형상은 고정, 유속만 X–Y 평면에서 회전).
    # 따라서 두 모드 모두 C·(x,y,z) = (z, −x, −y) 하나로 처리된다.
    def _disp_xyz(xs, ys, zs):
        """원좌표 리스트 → 표시좌표. None(선분 끊기)은 그대로 통과."""
        if not display_frame:
            return xs, ys, zs
        return ([v for v in zs],
                [None if v is None else -v for v in xs],
                [None if v is None else -v for v in ys])

    if display_frame:
        pts = np.column_stack([pts[:, 2], -pts[:, 0], -pts[:, 1]])

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ii, ji, ki = faces[:, 0], faces[:, 1], faces[:, 2]

    # ── 전체 범위 계산 ────────────────────────────────────────────────────
    cx = (float(x.min()) + float(x.max())) / 2
    cy = (float(y.min()) + float(y.max())) / 2
    cz = (float(z.min()) + float(z.max())) / 2
    span = max(float(x.max()) - float(x.min()),
               float(y.max()) - float(y.min()),
               float(z.max()) - float(z.min()))

    # ── 유속 방향 화살표 ──────────────────────────────────────────────────
    # 표시 프레임의 유속 단위벡터 = R·(솔버 유속) = (sinα, −cosα, 0) — 두 모드 공통.
    # (원좌표 모드에서는 종전대로 X–Z 평면 화살표를 유지한다.)
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    if display_frame:
        fu = (sin_t, -cos_t, 0.0)
    else:
        fu = (cos_t, 0.0, sin_t)
    aLen  = span * 0.55
    a_s = (cx - fu[0]*aLen,        cy - fu[1]*aLen,        cz - fu[2]*aLen)
    a_e = (cx + fu[0]*aLen*0.25,   cy + fu[1]*aLen*0.25,   cz + fu[2]*aLen*0.25)

    # ── Figure 조립 ───────────────────────────────────────────────────────
    fig = go.Figure()

    # STL 메쉬
    _mesh_kw = dict(
        x=x, y=y, z=z, opacity=0.78, flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.2, roughness=0.5),
        lightposition=dict(x=1, y=2, z=3),
        showlegend=False, hoverinfo='skip',
    )
    if highlight_surface:
        # 시스템이 계산한 표면적(compute_surface_area)에 실제로 포함된 영역만 녹색.
        # 그 값은 STL 파일 1개분이므로, 타일링(Nx×Ny) 시에도 기준 타일 하나만
        # 녹색이 되고 복제 타일은 기본색으로 남는다 → '무엇을 센 면적인지'가 보인다.
        _m_hi = len(base_faces)
        fig.add_trace(go.Mesh3d(
            i=ii[:_m_hi], j=ji[:_m_hi], k=ki[:_m_hi],
            color='#2ecc71', **_mesh_kw))
        if len(ii) > _m_hi:      # 복제 타일(면적 집계 대상 아님)
            fig.add_trace(go.Mesh3d(
                i=ii[_m_hi:], j=ji[_m_hi:], k=ki[_m_hi:],
                color='#5dade2', **_mesh_kw))
    else:
        fig.add_trace(go.Mesh3d(i=ii, j=ji, k=ki, color='#5dade2', **_mesh_kw))

    # 단위셀 경계 박스 (타일이 2개 이상일 때)
    if do_tile:
        for ix, iy in offsets:
            dx, dy = ix * step_x, iy * step_y
            # 상·하 사각형 + 수직 연결선 (None으로 선분 끊기)
            ex = [bx_min+dx, bx_max+dx, bx_max+dx, bx_min+dx, bx_min+dx, None,
                  bx_min+dx, bx_max+dx, bx_max+dx, bx_min+dx, bx_min+dx, None,
                  bx_min+dx, bx_min+dx, None, bx_max+dx, bx_max+dx, None,
                  bx_max+dx, bx_max+dx, None, bx_min+dx, bx_min+dx]
            ey = [by_min+dy, by_min+dy, by_max+dy, by_max+dy, by_min+dy, None,
                  by_min+dy, by_min+dy, by_max+dy, by_max+dy, by_min+dy, None,
                  by_min+dy, by_min+dy, None, by_min+dy, by_min+dy, None,
                  by_max+dy, by_max+dy, None, by_max+dy, by_max+dy]
            ez = [bz_min]*5 + [None] + [bz_max]*5 + [None,
                  bz_min, bz_max, None, bz_min, bz_max, None,
                  bz_min, bz_max, None, bz_min, bz_max]
            _ex, _ey, _ez = _disp_xyz(ex, ey, ez)
            fig.add_trace(go.Scatter3d(
                x=_ex, y=_ey, z=_ez,
                mode='lines',
                line=dict(color='#f39c12', width=1),
                showlegend=False, hoverinfo='skip',
            ))

    # 유속 화살표 shaft (표시 프레임 기준으로 이미 계산된 a_s → a_e)
    fig.add_trace(go.Scatter3d(
        x=[a_s[0], a_e[0]], y=[a_s[1], a_e[1]], z=[a_s[2], a_e[2]],
        mode='lines',
        line=dict(color='#e74c3c', width=6),
        showlegend=False, hoverinfo='skip',
    ))
    # 유속 화살표 cone
    fig.add_trace(go.Cone(
        x=[a_e[0]], y=[a_e[1]], z=[a_e[2]],
        u=[fu[0] * aLen * 0.22], v=[fu[1] * aLen * 0.22], w=[fu[2] * aLen * 0.22],
        colorscale=[[0, '#e74c3c'], [1, '#e74c3c']],
        showscale=False,
        sizemode='absolute', sizeref=span * 0.1,
        showlegend=False, hoverinfo='skip',
    ))

    # ── 모드 정보 · 유속 라벨을 annotation으로 배치 (겹침 없음) ─────────
    mode_label = "단위 셀 모드" if mode == "unit_cell" else "전체 구조 모드"
    tile_label = f"  ({nx}×{ny} 타일)" if do_tile else ""

    # 첫 렌더에만 camera를 넣고, 이후 위젯 변경 렌더에서는 빼서 uirevision이
    # 사용자의 마우스 카메라를 보존하게 한다(render_field_plotly의 (D) 주석 참고).
    # 축 제목: 표시 프레임이면 유동장 뷰와 같은 의미를 함께 적는다.
    # 제목이 길면 3D 씬 가장자리에서 잘리므로 의미는 유지하되 짧게 적는다.
    _ax_t = (("X [mm] · 법선", "Y [mm] · 유속", "Z [mm] · 수심")
             if display_frame else ("X [mm]", "Y [mm]", "Z [mm]"))
    # 축 가독성: 종전엔 연한 하늘색 배경 위에 gridcolor="white" 라 격자가 거의
    # 안 보였고, 눈금·제목 폰트 색을 지정하지 않아 기본 연회색으로 나왔다.
    # 배경은 더 밝게, 격자·축선·글자는 진하게 해서 대비를 확보한다.
    def _axis(title, bg):
        return dict(
            title=dict(text=title, font=dict(size=12, color="#0d2d4e")),
            tickfont=dict(size=11, color="#123a63"),
            backgroundcolor=bg, showbackground=True,
            gridcolor="#8fb0cf", gridwidth=1,
            zeroline=True, zerolinecolor="#456d94", zerolinewidth=2,
            showline=True, linecolor="#456d94", linewidth=2,
            tickcolor="#456d94", ticklen=4, ticks="outside",
        )

    _scene = dict(
        xaxis=_axis(_ax_t[0], "#f4f9fd"),
        yaxis=_axis(_ax_t[1], "#f4f9fd"),
        zaxis=_axis(_ax_t[2], "#e9f1f9"),
        aspectmode='data',
        bgcolor='rgba(250,253,255,1)',
    )
    if init_camera:
        # 종전 (1.4,1.0,0.9) 은 플롯 박스가 씬을 가득 채워 축 제목이 잘렸다.
        # 시점을 약간 뒤로 물려 가장자리에 제목이 들어갈 여유를 만든다.
        _scene['camera'] = dict(eye=dict(x=1.62, y=1.16, z=1.04))

    fig.update_layout(
        showlegend=False,
        annotations=[
            # 좌상단 모드 배지
            dict(
                text=f"<b>{mode_label}</b>{tile_label}",
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=12, color="#1a4a8a"),
                bgcolor="rgba(255,255,255,0.80)",
                bordercolor="#1a4a8a", borderwidth=1, borderpad=5,
            ),
            # 좌하단 유속 라벨
            dict(
                text=f"→ 유속 방향  α = {angle_deg:.0f}°",
                xref="paper", yref="paper",
                x=0.01, y=0.03,
                xanchor="left", yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#e74c3c"),
                bgcolor="rgba(255,255,255,0.80)",
                borderpad=4,
            ),
        ],
        scene=_scene,
        uirevision='stlpreview',
        # 여백 0 이면 3D 씬 가장자리의 축 제목·눈금이 잘린다.
        margin=dict(l=12, r=12, t=12, b=12),
        height=470,
        paper_bgcolor='#f0f8ff',
    )
    return fig


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

    # ─── 프로젝트 (항목3·4) ───────────────────────────────────────────────
    st.markdown("### 📁 프로젝트")
    _active_proj = ss.get("active_project")
    st.caption(f"현재 프로젝트: **{_active_proj}**" if _active_proj
               else "현재 프로젝트: _(없음 — 임시 작업)_")
    with st.expander("저장 / 불러오기 / 새로 만들기", expanded=True):
        # v13 항목7: 새 프로젝트 생성/불러오기 직후 활성 프로젝트명을 이름 칸에
        # 표시한다. 위젯 key 가 이미 세션에 있으면 value= 는 무시되므로,
        # 활성 프로젝트가 바뀌면 위젯 생성 전에 세션 키를 직접 동기화한다.
        if _active_proj and ss.get("_last_synced_proj_name") != _active_proj:
            ss["proj_name_input"] = _active_proj
            ss["_last_synced_proj_name"] = _active_proj
        _pname = st.text_input("프로젝트 이름",
                               key="proj_name_input",
                               placeholder="예: onemesh2_기본")

        # v13 항목3·8: 저장은 on_click 콜백으로 — 본문 내 st.rerun() 은 스크립트를
        # 조기 중단시켜 그 아래에서 생성되는 위젯(유속/영각 단계, 표시 방식 등)의
        # 상태를 Streamlit 이 청소 → 결과 분석 탭 매트릭스·시각화가 기본값으로
        # 리셋되던 버그(항목8). 콜백은 다음 런 시작 전에 실행되므로 전체 위젯이
        # 정상 렌더되고 결과·CSV·매트릭스가 그대로 유지된다.
        def _do_save_project():
            _nm = (ss.get("proj_name_input") or "").strip()
            if _nm:
                save_project(mode, _nm)
                ss["_save_toast"] = _nm
            else:
                ss["_save_toast_warn"] = True
        st.button("💾 저장", use_container_width=True, type="primary",
                  key="proj_save_btn", on_click=_do_save_project)
        if ss.pop("_save_toast", None):
            st.success(f"프로젝트 저장됨 · 결과/CSV/매트릭스 유지")
        if ss.pop("_save_toast_warn", None):
            st.warning("프로젝트 이름을 입력하세요.")

        _projs = list_project_names(mode)
        if _projs:
            # 버그: 새 이름으로 저장해도 이 드롭다운은 이전 선택을 그대로 들고 있어,
            # 저장 직후 '불러오기'를 누르면 엉뚱한 프로젝트가 로드됐다("저장 후
            # 불러오기가 안 된다"의 실체). 활성 프로젝트가 바뀌면 위젯 생성 '전에'
            # 선택값을 동기화한다(이름 칸 동기화와 같은 패턴).
            if (_active_proj in _projs
                    and ss.get("_last_synced_load_sel") != _active_proj):
                ss["proj_load_sel"] = _active_proj
                ss["_last_synced_load_sel"] = _active_proj
            st.selectbox("불러올 프로젝트", options=_projs, key="proj_load_sel")
            # v13 항목2·9: 불러오기도 콜백 — _pending_load_project 로 넘기면
            # 다음 런 시작 시(위젯 생성 전) _apply_project_load 가 세션 상태를
            # 갱신하고, active_project 설정으로 매트릭스·CSV·결과가 그 프로젝트
            # 폴더 기준으로 자동 복원된다.
            def _do_load_project():
                # 항목5: 다른 프로젝트로 바꾸기 전 미저장 변경 확인
                if _project_has_unsaved_changes(mode):
                    ss["_pending_load_after_guard"] = ss.get("proj_load_sel")
                    ss["_load_guard_requested"] = True
                else:
                    ss._pending_load_project = ss.get("proj_load_sel")
                    ss["_last_synced_proj_name"] = None   # 이름 칸 재동기 유도
            # 불러오기 버튼은 하나만 둔다(종전의 '파일에서 열기…' 다이얼로그는
            # 이 드롭다운과 기능이 겹쳐 화면만 산만하게 만들어 제거).
            st.button("📂 불러오기", use_container_width=True, type="primary",
                      key="proj_load_btn", on_click=_do_load_project)
        else:
            st.caption("저장된 프로젝트가 없습니다.")

        def _do_new_project():
            # v13 항목5: 새 프로젝트 진입 전 미저장 변경 감지 → 있으면 확인
            # 다이얼로그, 없으면 곧바로 새 프로젝트 대화상자.
            ss["_new_proj_requested"] = True
        st.button("🆕 새 프로젝트", use_container_width=True,
                  key="proj_new_btn", on_click=_do_new_project)
    # 새 프로젝트 요청 처리 — 미저장 변경 가드
    if ss.pop("_new_proj_requested", False):
        if _project_has_unsaved_changes(mode):
            _unsaved_guard_dialog("new")
        else:
            _new_project_dialog()
    # 불러오기 요청 처리 — 미저장 변경 가드
    if ss.pop("_load_guard_requested", False):
        _unsaved_guard_dialog("load")

    # v13 항목1: 새로고침(F5/Ctrl+Shift+R) 가로채기 — 미저장 변경이 있으면
    # 브라우저 표준 '나가시겠습니까?' 경고를 띄운다. 결과·조건은 프로젝트
    # 폴더 + .input_state.json 에 저장돼 새로고침 후 자동 복구되지만(위 참조),
    # 저장 안 한 변경은 이 경고로 사용자에게 알린다. active_project 유무를
    # 부모 윈도우 플래그로 전달한다.
    _dirty_flag = "1" if _project_has_unsaved_changes(mode) else "0"
    import streamlit.components.v1 as _cvbu
    _cvbu.html(f"""<script>
(function(){{
  var W=window.parent;
  W.__cfdDirty="{_dirty_flag}";
  if(!W.__cfdBeforeUnload){{
    W.__cfdBeforeUnload=function(e){{
      if(W.__cfdDirty==="1"){{
        e.preventDefault(); e.returnValue=""; return "";
      }}
    }};
    W.addEventListener("beforeunload", W.__cfdBeforeUnload);
  }}
}})();
</script>""", height=0)
    st.divider()

    # ─── 공통 물리 조건 ───────────────────────────────────────────────────
    st.markdown("### 🌊 물리 조건")
    # 유체 프리셋(요구서 §7) — 선택 시 ρ·ν 를 채운다. '사용자 정의'면 손대지 않는다.
    _fl = st.selectbox("유체 종류", ["해수 (20℃)", "담수 (20℃)", "공기 (20℃)", "사용자 정의"],
                       key="fluid_type",
                       help="선택하면 밀도와 동점성계수가 자동으로 채워집니다. "
                            "값을 직접 바꾸려면 '사용자 정의'를 고르십시오.")
    if _fl in FLUID_PRESETS and ss.get("_fluid_applied") != _fl:
        ss["rho"] = float(FLUID_PRESETS[_fl]["rho"])
        ss["nu"]  = float(FLUID_PRESETS[_fl]["nu_e6"])
        ss["_fluid_applied"] = _fl
    elif _fl == "사용자 정의":
        ss["_fluid_applied"] = _fl
    # 범위: 담수·공기까지 담기 위해 넓힌다(기본값은 종전과 동일한 해수 값).
    rho = st.number_input("밀도 ρ [kg/m³]", value=1025.0,
                          min_value=0.5, max_value=1200.0, step=1.0, key="rho")
    nu  = st.number_input("동점성계수 ν [×10⁻⁶ m²/s]",
                          value=1.19, min_value=0.05, max_value=30.0, step=0.01, key="nu")
    ti  = st.slider("난류 강도 I [%]", 1, 20, 5, key="ti")
    st.divider()

    # ─── 시스템 설정 ──────────────────────────────────────────────────────
    st.markdown("### 🖥️ 시스템 설정")
    max_cores = get_cpu_count()
    n_cores = st.slider("MPI 코어 수", 1, max_cores, max_cores,
                        help=f"물리 코어 수: {max_cores} (하이퍼스레딩 제외 — 초과 설정 시 성능 저하)")
    ss.n_cores = n_cores

    refresh_sec = st.slider("시각화 갱신 주기 [초]", 2, 30, 3,
                            help="짧을수록 실시간에 가깝지만 CPU 사용량 증가")
    ss.refresh_interval = refresh_sec
    st.divider()

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

# 이 스크립트 실행이 '무슨 상태로 화면을 그렸는지' 캡처한다.
# 백그라운드 스레드가 렌더 도중 job_status를 running→done으로 바꾸면, 맨 끝
# 자동 새로고침 시점엔 이미 done이라 재실행이 안 돼 화면이 직전 진행률(예: 96.8%)
# 에 얼어붙는다. 이 값으로 '방금 끝났는지'를 판정해 마지막 한 번 더 그린다.
_status_at_render = ss.job_status

# 경과 시간 + 동적 남은시간 추정(항목1)
_elapsed_str = ""
_eta_str = ""
if ss.job_status == "running" and ss.get("job_start_time"):
    _elapsed_sec = time.time() - ss.job_start_time
    _elapsed_str = f"  &nbsp;&nbsp;⏱️ 경과: **{fmt_elapsed(_elapsed_sec)}**"
    # 항목1: 실행 중 실제 경과·진행률로 남은시간을 동적 추정해 계속 갱신한다.
    # 진행률이 충분히(≥3%) 쌓이면 실측 기반(remaining = 경과×(100-p)/p)으로,
    # 그 전에는 시작 시 저장한 정적 추정에서 경과를 뺀 값으로 표시(초기 불안정 완화).
    _p = float(ss.progress)
    if _p >= 3.0:
        _rem = _elapsed_sec * (100.0 - _p) / _p
        _src = "실측"
    else:
        _rem = max(0.0, float(ss.get("est_total_min", 0.0)) * 60.0 - _elapsed_sec)
        _src = "추정"
    _eta_str = f"  &nbsp;&nbsp;⏳ 예상 남은: **{fmt_elapsed(_rem)}** ({_src})"

st.markdown(
    f"**해석 모드:** {'🔬 단위 셀 (Unit Cell)' if mode=='unit_cell' else '🏗️ 전체 구조 (Full Structure)'}  "
    f"&nbsp;&nbsp;**상태:** {status_badge(ss.job_status)}  "
    f"&nbsp;&nbsp;**진행률:** {ss.progress:.1f}%"
    f"{_elapsed_str}{_eta_str}",
    unsafe_allow_html=True
)

# 새로고침/세션 단절 후 복구된 작업을 보고 있을 때 안내
if ss.get("_detached_view"):
    st.info(
        "🔄 **새로고침으로 복구된 화면입니다.** 해석은 백그라운드에서 계속 진행 중이며, "
        "진행률·로그는 디스크에서 자동으로 불러와 갱신됩니다. "
        "단, 이 화면에서는 '중지' 버튼이 동작하지 않을 수 있습니다.",
        icon="ℹ️",
    )

# ─── 진행률 바 + 마지막 로그 라인 실시간 표시 ────────────────────────────
# 항목3: st.progress 는 0.0~1.0 만 허용. 진행률 계산(케이스별 pct 등)이 라운딩·
# 타이밍·인덱싱으로 음수(-0.006 등)나 1 초과가 되면 StreamlitAPIException 으로
# 해석 화면이 통째로 죽는다. 모든 진행값을 [0,1] 로 클램프해 방지한다.
def _clamp01(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

if ss.job_status == "running":
    _bar_text = f"⚙️ {ss.current_step}" if ss.current_step else "⚙️ 해석 진행 중..."
    st.progress(_clamp01(ss.progress / 100),
                text=f"전체 진행률 {ss.progress:.1f}%  |  {_bar_text[2:].strip()}")
    # 배치 케이스별 진행률
    _cp = ss.get("case_progress", {})
    if _cp and _cp.get("total", 1) > 1:
        _cp_pct = float(_cp.get("pct", 0.0))
        _cp_idx = _cp.get("case_idx", 1)
        _cp_tot = _cp.get("total", 1)
        st.progress(
            _clamp01(_cp_pct / 100),
            text=f"케이스 {_cp_idx}/{_cp_tot} 진행률: {max(0.0, _cp_pct):.1f}%",
        )
    # 마지막 유효 로그 라인 강조 표시
    _last_logs = [l for l in (ss.log_lines or []) if l.strip()]
    if _last_logs:
        _last = _last_logs[-1]
        _log_color = "#d93025" if "❌" in _last or "오류" in _last else "#1a73e8"
        st.markdown(
            f'<div style="font-size:12px;color:{_log_color};'
            f'background:#f8f9fa;padding:4px 10px;border-left:3px solid {_log_color};'
            f'border-radius:4px;margin:2px 0;font-family:monospace;">'
            f'📋 {_last[:120]}</div>',
            unsafe_allow_html=True
        )
elif ss.job_status == "done":
    st.progress(1.0, text="✅ 해석 완료!")
elif ss.job_status == "error":
    st.error(f"❌ 오류 발생: {ss.current_step}")

# ─── 파이프라인 단계 패널 ─────────────────────────────────────────────────
_pipeline = ss.get("pipeline_steps", [])
if _pipeline and ss.job_status in ("running", "done", "error"):
    _STATUS_ICON = {
        "pending": "⬜",
        "running": "🔵",
        "done":    "✅",
        "error":   "❌",
    }
    _STATUS_COLOR = {
        "pending": "#888",
        "running": "#1a73e8",
        "done":    "#0f9d58",
        "error":   "#d93025",
    }
    _pcols = st.columns(len(_pipeline))
    for _col, _step in zip(_pcols, _pipeline):
        _icon  = _STATUS_ICON.get(_step["status"], "⬜")
        _color = _STATUS_COLOR.get(_step["status"], "#888")
        _pct   = float(_step["pct"])          # 소수점 진행률 유지
        _detail = _step.get("detail", "")
        with _col:
            st.markdown(
                f"<div style='text-align:center;padding:6px;border:1px solid {_color};"
                f"border-radius:8px;background:{'#e8f4fd' if _step['status']=='running' else '#f8f9fa'}'>"
                f"<div style='font-size:20px'>{_icon}</div>"
                f"<div style='font-size:12px;font-weight:600;color:{_color}'>{_step['label']}</div>"
                f"<div style='font-size:11px;color:#555'>{_detail or _step['status']}</div>"
                f"<div style='background:#e0e0e0;border-radius:4px;height:6px;margin-top:4px'>"
                f"<div style='background:{_color};width:{_pct:.1f}%;height:6px;border-radius:4px'></div></div>"
                f"<div style='font-size:10px;color:{_color}'>{_pct:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ─── 탭 구성 ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _start_single_analysis(
    mode, speed, angle,
    cell_size, cage_d, cage_h,
    end_time, n_cores, rho, ti,
    nx=1, ny=1,
    residual_control=1e-4,
    write_interval=100,
    refine_level=3,
    solidity=None,
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
    # 사용자 지정 Aref 는 스레드 시작 전(메인 스레드)에 값으로 확정해 클로저로 넘긴다.
    _aref_ovr = _effective_aref()
    if _aref_ovr > 0:
        add_log(f"기준면적 Aref = {_aref_ovr:.6e} m² (사용자 직접 입력)")

    if mode == "unit_cell" and not stl_net:
        st.error("❌ 그물 STL 파일을 먼저 업로드하세요.")
        return

    set_status("running", "케이스 준비 중...")
    ss.progress       = 0.0
    ss.log_lines      = []
    ss.job_start_time = time.time()
    ss.refresh_count  = 0
    add_log(f"해석 시작: 모드={mode}, U={speed}m/s, α={angle}°")

    # ── 파이프라인 단계 초기화 ──────────────────────────────────────────────
    _PIPE_LABELS = [
        "blockMesh",
        "surfaceFeatureExtract",
        "snappyHexMesh",
        "simpleFoam",
        "reconstructPar",
    ]
    ss.pipeline_steps = [
        {"label": lbl, "status": "pending", "pct": 0, "detail": ""}
        for lbl in _PIPE_LABELS
    ]

    # 케이스 디렉토리 생성
    case_name = f"{mode}_U{speed:.2f}_A{angle:.1f}_{datetime.now():%H%M%S}"
    case_dir  = RESULTS_DIR / mode / case_name
    ss.last_result_dir = str(case_dir)

    def _run():
        # ── step_cb: 단계별 진행 상태 업데이트 ─────────────────────────────
        def _step_cb(label: str, status: str, pct: int, detail: str):
            steps = ss.get("pipeline_steps", [])
            for step in steps:
                # _run_step label 예: "배경 격자 생성 (blockMesh)" — 키워드 부분 매칭
                if step["label"] in label or label in step["label"]:
                    step["status"] = status
                    step["pct"]    = pct
                    step["detail"] = detail
                    break
            ss.pipeline_steps = steps
            if status == "running":
                ss.current_step = f"{label}: {detail}" if detail else label

        try:
            # 케이스 빌드
            if mode == "unit_cell":
                builder = UnitCellCaseBuilder(
                    case_dir=case_dir,
                    stl_path=stl_net,
                    speed=speed, angle_deg=angle,
                    cell_size=cell_size or None,
                    n_cores=n_cores,
                    nx=nx, ny=ny,
                    residual_control=residual_control,
                    end_time=end_time,
                    write_interval=write_interval,
                    refine_level=refine_level,
                    solidity=solidity,
                    aref_override=_aref_ovr,
                    rho=float(ss.get("rho", 1025.0)),
                    nu=float(ss.get("nu", 1.19)) * 1e-6,
                )
            else:
                builder = FullStructureCaseBuilder(
                    case_dir=case_dir,
                    cage_stl=stl_cage,
                    net_stl=stl_net,
                    speed=speed, angle_deg=angle,
                    cage_diameter=cage_d or 10.0,
                    cage_depth=cage_h or 5.0,
                    n_cores=n_cores,
                    residual_control=residual_control,
                    end_time=end_time,
                    write_interval=write_interval,
                    aref_override=_aref_ovr,
                    rho=float(ss.get("rho", 1025.0)),
                    nu=float(ss.get("nu", 1.19)) * 1e-6,
                )
            builder.build()
            add_log("✅ 케이스 빌드 완료")

            # 해석 실행
            runner = OpenFOAMRunner(
                case_dir=case_dir,
                n_cores=n_cores,
                progress_cb=lambda p, s, e: (
                    setattr(ss, "progress", p),
                    setattr(ss, "current_step", f"반복 {s}/{e}회  ← 현재/최대")
                ),
                log_cb=add_log,
                step_cb=_step_cb,
            )
            ss.job_runner = runner

            if not runner.run_blockMesh():
                raise RuntimeError("blockMesh 실패 — logs/ 폴더의 로그를 확인하세요.")
            set_status("running", "snappyHexMesh 실행 중...")
            runner.run_surfaceFeatureExtract()
            if not runner.run_snappyHexMesh():
                raise RuntimeError("snappyHexMesh 실패 — logs/ 폴더의 로그를 확인하세요.")
            set_status("running", "CFD 해석 중...")
            if not runner.run_solver(end_time=end_time):
                raise RuntimeError("simpleFoam 실패 — logs/ 폴더의 로그를 확인하세요.")
            runner.run_reconstructPar()

            # 결과 추출
            extractor = ResultExtractor(case_dir, speed, angle, rho)
            csv_out   = RESULTS_DIR / mode / f"results_{mode}.csv"
            extractor.save_csv(csv_out)
            ss.results_csv = str(csv_out)

            ss.progress = 100.0
            set_status("done", "해석 완료!")
            add_log(f"✅ 해석 완료! 결과: {csv_out}")

        except Exception as e:
            set_status("error", str(e))
            add_log(f"❌ 오류: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    # 백그라운드 스레드에서도 Streamlit session_state(ss)에 접근할 수 있도록
    # 현재 스크립트 실행 컨텍스트를 스레드에 부착한다. (이 호출이 없으면
    # 스레드 내부의 ss 접근이 NoSessionContext로 실패해 해석이 조용히 멈춘다)
    if add_script_run_ctx is not None:
        add_script_run_ctx(thread)
    ss.job_thread = thread
    thread.start()
    st.rerun()


def _scan_done_conditions(mode):
    """프로젝트(모드)에서 이미 완료된 (유속, 영각) 조건 집합을 반환.
    항목9 덮어쓰기 경고용 — '해석완료_' 접두어 + 실제 결과(시간 디렉토리) 보유 기준."""
    import re as _re_dc
    _done = set()
    # 항목2: 활성 프로젝트가 있으면 그 폴더만 검사 → 새 프로젝트(빈 폴더)는 완료
    # 조건이 없어 덮어쓰기 경고가 뜨지 않는다.
    _root = _results_root(mode)
    if not _root.exists():
        return _done
    for _d in _root.iterdir():
        if not _d.is_dir() or not _d.name.startswith("해석완료_"):
            continue
        _m = _re_dc.search(r"_U([0-9.]+)_A(-?[0-9.]+)", _d.name)
        if not _m:
            continue
        try:
            _tdirs = [t for t in _d.iterdir()
                      if t.is_dir() and t.name not in ("0",)
                      and t.name.replace(".", "", 1).isdigit()]
        except Exception:
            _tdirs = []
        if _tdirs:
            _done.add((round(float(_m.group(1)), 2), round(float(_m.group(2)), 1)))
    return _done


def _start_mesh_independence(mode, levels, n_cores, rho, ti):
    """격자 독립성 시험을 백그라운드로 실행한다(요구서 §19).

    통상 배치와 같은 실행 경로(BatchAnalysisManager)를 레벨마다 한 번씩 쓴다.
    """
    stl_net = Path(ss.stl_net_path) if ss.get("stl_net_path") else None
    if not stl_net or not stl_net.exists():
        st.error("STL 을 먼저 업로드하세요."); return
    stl_paths = {"net": stl_net}
    if mode == "full_structure" and ss.get("stl_cage_path"):
        stl_paths["cage"] = Path(ss.stl_cage_path)
    _speed = float(ss.get("u_min", 1.0))
    _angle = float(ss.get("a_min", 0.0))
    params = {
        "n_cores": n_cores, "nx": ss.get("unit_nx", 1), "ny": ss.get("unit_ny", 1),
        "end_time": int(ss.get("end_time_preset", 2000)),
        "residual_control": float(ss.get("residual_preset", "1e-4")),
        "write_interval": int(ss.get("write_interval_preset", 100)),
        "aref_override": _effective_aref(),
        "rho": float(ss.get("rho", 1025.0)), "nu": _nu_si(),
    }
    if mode == "full_structure":
        params["cage_diameter"] = float(ss.get("cage_d", 10.0))
        params["cage_depth"] = float(ss.get("cage_h", 5.0))
        if ss.get("net_grid_redesign"):
            params["net_grid_redesign"] = True
            params["net_grid_target_cells"] = float(ss.get("net_grid_target_cells", 75.0))
    ss.log_lines = []
    ss.job_start_time = time.time()
    set_status("running", f"격자 독립성 시험 ({len(levels)}개 레벨)")
    add_log(f"🔬 격자 독립성 시험 시작 — 레벨 {levels}, U={_speed} m/s, α={_angle}°")
    _root = _results_root(mode) / f"mesh_independence_{datetime.now():%y%m%d_%H%M}"

    def _run():
        try:
            rows = run_mesh_independence(
                mode=mode, stl_paths=stl_paths, speed=_speed, angle=_angle,
                levels=levels, common_params=params, results_root=_root,
                log_cb=add_log,
                progress_cb=lambda p, s, e, label="": (
                    setattr(ss, "progress", max(0.0, min(100.0, float(p)))),
                    setattr(ss, "current_step", label or f"레벨 {s}/{e}")))
            ss["mi_rows"] = rows
            ss["mi_done"] = any(r.get("Cd") is not None for r in rows)
            set_status("done", "격자 독립성 시험 완료")
            add_log("✅ 격자 독립성 시험 완료")
        except Exception as e:
            set_status("error", str(e)); add_log(f"❌ 격자 독립성 시험 오류: {e}")

    th = threading.Thread(target=_run, daemon=True)
    if add_script_run_ctx is not None:
        add_script_run_ctx(th)
    ss.job_thread = th
    th.start()


def _start_reference_case(key, speed, n_cores, rho, ti):
    """문헌값이 있는 기본 형상을 같은 파이프라인으로 실행한다(요구서 §20)."""
    rc = REFERENCE_CASES[key]
    stl = None
    if rc.get("stl") and (APP_DIR / rc["stl"]).exists():
        stl = APP_DIR / rc["stl"]
    elif key == "cylinder":
        stl = STL_UPLOAD_DIR / "reference_cylinder.stl"
        if not stl.exists():
            write_cylinder_stl(stl, diameter_mm=rc["length_m"]*1000.0,
                               length_mm=rc["length_m"]*1000.0*4)
    if not stl or not Path(stl).exists():
        st.error(f"검증용 STL 을 찾을 수 없습니다: {rc.get('stl')}"); return

    params = {
        "n_cores": n_cores,
        "end_time": int(ss.get("end_time_preset", 2000)),
        "residual_control": float(ss.get("residual_preset", "1e-4")),
        "write_interval": int(ss.get("write_interval_preset", 100)),
        "refine_level": int(ss.get("refine_level_preset", 3)),
        "auto_refine": True,
        "rho": float(ss.get("rho", 1025.0)), "nu": _nu_si(),
    }
    ss.log_lines = []
    ss.job_start_time = time.time()
    set_status("running", f"검증 케이스 실행 — {rc['label']}")
    add_log(f"🔬 검증 케이스 시작: {rc['label']} · U={speed} m/s")
    _root = _results_root("full_structure") / f"reference_{key}_{datetime.now():%y%m%d_%H%M}"

    def _run():
        try:
            mgr = BatchAnalysisManager(
                mode="full_structure", stl_paths={"net": Path(stl)},
                speeds=[float(speed)], angles=[0.0],
                output_csv=_root / "force_coeffs.csv", common_params=params,
                progress_cb=lambda p, s, e, label="": (
                    setattr(ss, "progress", max(0.0, min(100.0, float(p)))),
                    setattr(ss, "current_step", label or "검증 케이스")),
                log_cb=add_log, results_root=_root)
            res = mgr.run_batch()
            if res and res[-1].get("Cd") is not None:
                cmp_ = reference_comparison(key, float(res[-1]["Cd"]),
                                            float(speed), _nu_si())
                ss["ref_result"] = cmp_
                ss["ref_done"] = True
                add_log(f"✅ 검증 완료 — CFD Cd={cmp_['Cd_cfd']:.4f} vs "
                        f"문헌 {cmp_['Cd_ref']} (오차 {cmp_['error_pct']:+.1f}%)")
                set_status("done", "검증 케이스 완료")
            else:
                set_status("error", "검증 케이스에서 Cd 를 얻지 못했습니다")
        except Exception as e:
            set_status("error", str(e)); add_log(f"❌ 검증 케이스 오류: {e}")

    th = threading.Thread(target=_run, daemon=True)
    if add_script_run_ctx is not None:
        add_script_run_ctx(th)
    ss.job_thread = th
    th.start()


def _start_batch_analysis(mode, speeds, angles, csv_path, n_cores, rho, ti, nx=1, ny=1):
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
    ss.progress       = 0.0
    ss.log_lines      = []
    ss.job_start_time = time.time()
    ss.refresh_count  = 0
    add_log(f"배치 해석 시작: {len(speeds)*len(angles)}개 케이스")

    # 계산량 프리셋(최소/보통/정밀)을 배치에도 반영
    _bp = {
        "n_cores": n_cores, "nx": nx, "ny": ny,
        "end_time":         int(ss.get("end_time_preset", 2000)),
        "refine_level":     int(ss.get("refine_level_preset", 3)),
        "residual_control": float(ss.get("residual_preset", "1e-4")),
        "write_interval":   int(ss.get("write_interval_preset", 100)),
        # 사용자 지정 기준면적(0 이면 빌더가 자동 계산)
        "aref_override":    _effective_aref(),
        # 유체 물성 — 종전에는 UI 입력이 케이스에 전혀 전달되지 않아
        # 템플릿 값(nu 1.19e-6, rhoInf 1025)이 항상 쓰였다.
        "rho":              float(ss.get("rho", 1025.0)),
        "nu":               float(ss.get("nu", 1.19)) * 1e-6,
    }
    # 가두리 치수: 전달하지 않으면 FullStructureCaseBuilder 기본값(10 m × 5 m)이
    # 쓰여 UI 입력이 무시된다 → 도메인 크기·lRef·Aref(D×H)가 모두 어긋난다.
    # 위젯 key("cage_d"/"cage_h")로 세션에서 직접 읽어 배치에도 반영한다.
    if mode == "full_structure":
        _bp["cage_diameter"] = float(ss.get("cage_d", 10.0))
        _bp["cage_depth"]    = float(ss.get("cage_h", 5.0))
        _bp["auto_refine"]   = bool(ss.get("auto_refine", True))
        if _bp["auto_refine"]:
            add_log("격자 자동 보정: 형상 최소두께 기준으로 정밀화 레벨을 상향합니다")
        # 그물 격자 설계 — 기본값(재설계 꺼짐 · 후류 자동)이면 종전 경로와 동일
        _bp["net_grid_redesign"] = bool(ss.get("net_grid_redesign", False))
        if _bp["net_grid_redesign"]:
            _bp["net_grid_target_cells"] = float(ss.get("net_grid_target_cells", 75.0))
            add_log(f"배경격자 재설계: 목표 {_bp['net_grid_target_cells']:.0f} "
                    "셀/임계치수 · 도메인 상류 2L·하류 5L·횡 ±2L")
        if ss.get("wake_box_mode") == "직접 지정":
            _bp["wake_box_level"] = int(ss.get("wake_box_level", 2))
            add_log(f"후류 정밀화 박스 레벨 직접 지정: {_bp['wake_box_level']}")
        add_log(f"가두리 치수: 직경 {_bp['cage_diameter']:.2f} m × "
                f"수심 {_bp['cage_depth']:.2f} m")

    # ── Solver 분기 (지시서 §4) — Steady 면 transient=None 으로 종전 경로 ──
    _tr_cfg = _transient_config()
    if _tr_cfg:
        _bp["transient"] = _tr_cfg
        add_log(f"⏱️ 비정상 해석(pimpleFoam): 물리시간 {_tr_cfg['end_time']:g}s · "
                f"deltaT {_tr_cfg['delta_t']:g} · maxCo {_tr_cfg['max_co']:g} · "
                f"PIMPLE({_tr_cfg['n_outer']},{_tr_cfg['n_correctors']},"
                f"{_tr_cfg['n_non_orth']}) · {_tr_cfg['turbulence']}")
        add_log("초기조건: " + ("simpleFoam 선행 수렴 "
                f"({_tr_cfg['steady_end_time']}회)" if _tr_cfg["init_from_steady"]
                else ("균일장 + 대칭 교란" if _tr_cfg.get("perturb") else "균일장")))
    else:
        add_log("해석 방식: 정상상태 (simpleFoam)")
    add_log(f"계산 조건: 반복 {_bp['end_time']} · 정밀화 {_bp['refine_level']} · "
            f"수렴 {_bp['residual_control']:.0e} · {n_cores}코어")
    if _bp["aref_override"] > 0:
        add_log(f"기준면적 Aref = {_bp['aref_override']:.6e} m² (사용자 직접 입력)")
    else:
        add_log("기준면적 Aref = 형상에서 자동 계산")

    # 항목1: 동적 ETA 의 초기(진행률<3%) 기준이 될 정적 총 예상시간을 저장.
    _n_cases = max(1, len(speeds) * len(angles))
    ss.est_total_min = estimate_total_minutes(
        mode, _n_cases, _bp["end_time"], _bp["refine_level"], n_cores)

    manager = BatchAnalysisManager(
        mode=mode,
        stl_paths=stl_paths,
        speeds=speeds,
        angles=angles,
        output_csv=csv_path,
        common_params=_bp,
        progress_cb=lambda p, s, e, label="": (
            setattr(ss, "progress", max(0.0, min(100.0, float(p)))),
            setattr(ss, "current_step",
                    label if label else f"케이스 {s}/{e}개  ← 완료/전체"),
            setattr(ss, "case_progress", {
                "case_idx": s, "total": e,
                # 항목3 원인: 전체진행률 p 가 (s-1)/e*100 보다 약간 작으면(라운딩·
                # 단계 경계 타이밍) (p-(s-1)/e*100)*e 가 음수가 되어 case pct < 0 →
                # st.progress(음수) 크래시. 0~100 으로 클램프해 원천 차단.
                "pct": round(min(max((p - (s - 1) / e * 100) * e, 0.0), 100.0), 2),
                "label": label,
            })
        ),
        log_cb=add_log,
        # 항목1: 활성 프로젝트가 있으면 케이스 디렉토리를 프로젝트 폴더 안에 둔다.
        results_root=_results_root(mode),
    )
    ss.batch_manager = manager

    def _run():
        try:
            _t0 = time.time()
            manager.run_batch()
            try:
                _ns = int(getattr(manager, "n_success", 0))
                _nf = int(getattr(manager, "n_failed", 0))
                # 항목3: 정직한 완료 상태. 성공 케이스가 0이면 '완료'가 아니라 실패로
                # 보고해야 한다(이전엔 전부 실패해도 '배치 해석 완료!'로 떠 사용자가
                # 결과가 있다고 오인). 성공 시간만 기록해 추정 보정 오염도 방지.
                _elapsed_min = (time.time() - _t0) / 60.0
                if _ns > 0:
                    record_case_minutes(
                        mode, _elapsed_min / _ns,
                        end_time=_bp.get("end_time"),
                        refine_level=_bp.get("refine_level"), n_cores=n_cores)
                    # 항목1 검증로그: 추정 vs 실측(전 워크플로 end-to-end)·오차율 표시.
                    _est = float(ss.get("est_total_min", 0.0))
                    if _est > 0:
                        _err = abs(_elapsed_min - _est) / _est * 100.0
                        add_log(f"⏱️ 추정 {_est:.1f}분 vs 실측 {_elapsed_min:.1f}분 "
                                f"(오차 {_err:.0f}%) — 모드 {mode}. 실측은 다음 추정에 보정 반영.")
                if _ns == 0:
                    set_status("error",
                               f"모든 케이스 실패 (0/{_n_cases} 완료) — 로그/형상을 확인하세요.")
                    add_log(f"❌ 배치 종료: 0/{_n_cases} 완료 (전부 실패). "
                            f"메싱/솔버 로그를 확인하세요.")
                elif _ns < _n_cases:
                    set_status("done", f"배치 부분 완료: {_ns}/{_n_cases} 성공 "
                                       f"({_nf} 실패)")
                    add_log(f"✅ 배치 부분 완료: {_ns}/{_n_cases} 성공, {_nf} 실패. CSV: {csv_path}")
                else:
                    set_status("done", f"배치 해석 완료! ({_ns}/{_n_cases})")
                    add_log(f"✅ 배치 완료! {_ns}/{_n_cases} CSV: {csv_path}")
            except Exception:
                pass
        except Exception as e:
            try:
                set_status("error", str(e))
                add_log(f"❌ 배치 오류: {e}")
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True)
    # 백그라운드 스레드에서도 Streamlit session_state(ss)에 접근할 수 있도록
    # 현재 스크립트 실행 컨텍스트를 스레드에 부착한다.
    if add_script_run_ctx is not None:
        add_script_run_ctx(thread)
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

def _auto_aref_info(mode: str) -> dict:
    """케이스 빌더가 '자동'으로 계산할 Aref[m²]와 산출 근거를 UI 표시용으로 반환.

    cfd_manager 의 빌더와 동일한 식을 쓰므로 여기 표시값 = 실제 controlDict 에
    들어갈 값이다(직접 입력을 켜지 않은 경우).
    반환 키: value(float|None), basis(str), open_shell(bool|None), surface(float|None)
    """
    out = {"value": None, "basis": "", "open_shell": None,
           "surface": None, "projected": None}
    if mode == "full_structure" and ss.get("stl_cage_path"):
        D = float(ss.get("cage_d", 10.0)); H = float(ss.get("cage_h", 5.0))
        out.update(value=D * H, basis=f"가두리 직경 D({D:.1f} m) × 수심 H({H:.1f} m) — STL 형상 미반영")
        return out
    net = ss.get("stl_net_path")
    if not (net and Path(net).exists()):
        out["basis"] = "STL 미업로드"
        return out
    try:
        from cfd_manager import (compute_projected_area, compute_surface_area,
                                 is_closed_surface)
        closed = is_closed_surface(Path(net))
        out["value"]      = compute_projected_area(Path(net), (0.0, 0.0, 1.0))
        out["open_shell"] = not closed
        out["surface"]    = compute_surface_area(Path(net))
        # 진짜 투영면적: 닫힌 표면은 ÷2 가 맞으므로 value 그대로,
        # 열린 곡면은 ÷2 가 과잉이므로 2 배 되돌린 값이 실제 투영면적이다.
        out["projected"]  = out["value"] if closed else out["value"] * 2.0
        out["basis"] = ("그물면 법선(Z축) 방향 투영면적 — 닫힌 표면" if closed
                        else "그물면 법선(Z축) 방향 투영면적 — 열린 곡면")
    except Exception as _e:
        out["basis"] = f"계산 실패 ({_e})"
    return out


def _transient_config() -> Optional[dict]:
    """Solver=Transient 일 때 pimpleFoam 설정 dict, Steady 면 None.

    None 을 반환하면 실행 경로가 종전 simpleFoam 과 완전히 동일해진다(회귀 방지).
    """
    if not str(ss.get("solver_mode", "")).startswith("Transient"):
        return None
    return {
        "end_time":         float(ss.get("tr_end_time", 30.0)),
        "delta_t":          float(ss.get("tr_delta_t", 1e-3)),
        "max_co":           float(ss.get("tr_max_co", 0.8)),
        "max_delta_t":      float(ss.get("tr_max_delta_t", 0.01)),
        "write_interval":   float(ss.get("tr_write_interval", 0.5)),
        "n_outer":          int(ss.get("tr_n_outer", 1)),
        "n_correctors":     int(ss.get("tr_n_corr", 2)),
        "n_non_orth":       int(ss.get("tr_n_non_orth", 0)),
        "turbulence":       str(ss.get("tr_turbulence", "kOmegaSST")),
        "init_from_steady": bool(ss.get("tr_init_steady", True)),
        "steady_end_time":  int(ss.get("tr_steady_iters", 1000)),
        "perturb":          bool(ss.get("tr_perturb", False)),
        "perturb_magnitude": float(ss.get("tr_perturb_mag", 0.01)),
        "avg_start":        float(ss.get("tr_avg_start", 0.0)) or None,
    }


def why(reason: str, label: str = "왜?") -> None:
    """자동 권장값의 근거를 펼쳐 보이는 공통 요소(요구서 §23).

    권장·경고를 표시하는 곳마다 같은 모양으로 붙여, 초보자가 '어디를 눌러야
    이유를 볼 수 있는지' 예측 가능하게 한다.
    """
    with st.expander(label, expanded=False):
        st.write(reason)


def _total_mem_gb() -> Optional[float]:
    """장비의 총 메모리[GB]. 사전 점검에서 예상 사용량과 비교한다."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return None


def _fmt_len(v_m: float) -> str:
    """길이를 사람이 읽기 쉬운 단위로 (1 m 이상은 m, 그 미만은 mm)."""
    return f"{v_m:.3f} m" if v_m >= 1.0 else f"{v_m*1000:.2f} mm"


def _re_length_m(mode: str) -> Tuple[float, str]:
    """Reynolds 수의 대표 길이[m]와 그 근거 문장(요구서 §8).

    사용자가 대표 길이를 바꿀 수 있어야 하므로 선택 결과를 반영한다.
    """
    _mode_sel = ss.get("re_length_mode", "자동")
    _stl = ss.get("stl_net_path") or ss.get("stl_cage_path")
    _spans = None
    if _stl and Path(_stl).exists():
        try:
            _spans = critical_dimension(Path(_stl)).get("spans")
        except Exception:
            _spans = None
    if _mode_sel == "직접 입력":
        v = float(ss.get("re_length_manual_mm", 0.0) or 0.0) / 1000.0
        if v > 0:
            return v, "사용자가 직접 입력한 값"
    if _mode_sel == "바운딩박스 최대변" and _spans:
        return max(_spans) / 1000.0, "STL 바운딩박스의 최대변"
    if _mode_sel == "임계 최소 치수" and _spans:
        return min(s for s in _spans if s > 0) / 1000.0, "STL 바운딩박스의 최소변(그물실 지름·판재 두께)"
    if mode == "unit_cell":
        return float(ss.get("cell_size_mm", 20.0)) / 1000.0, "단위 셀 한 변(망목)"
    return float(ss.get("cage_d", 10.0)), "가두리 직경"


def _nu_si() -> float:
    """물리 조건에서 입력한 동점성계수 ν [m²/s]. UI 단위는 ×10⁻⁶ m²/s.

    종전에는 Reynolds 수 표시가 1.19e-6 을 하드코딩해 사용자 입력을 무시했다.
    """
    try:
        v = float(ss.get("nu", 1.19)) * 1e-6
    except (TypeError, ValueError):
        return 1.19e-6
    return v if v > 0 else 1.19e-6


def _effective_aref() -> float:
    """해석에 실제로 쓸 Aref[m²]. '직접 입력' 모드이고 값이 양수일 때만 값을 반환,
    그 외에는 0.0 (= 빌더가 자동 계산)."""
    if ss.get("aref_mode") == "직접 입력":
        try:
            v = float(ss.get("aref_manual_m2") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if v > 0:
            return v
    return 0.0


tab_input, tab_results, tab_help = st.tabs([
    "📂 입력 설정",
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

                # STL 자동 감지
                from cfd_manager import detect_stl_cell_size
                _info = detect_stl_cell_size(Path(ss.stl_net_path))
                ss.auto_cell_size_mm  = _info["cell_size_mm"]
                ss.auto_wire_d_mm     = _info["wire_diameter_mm"]
                ss.auto_solidity      = _info["solidity"]
                # 위젯 기본값 업데이트 (다음 렌더링에 반영)
                ss.cell_size_mm       = _info["cell_size_mm"]
                ss.solidity_input     = _info["solidity"]

                _front_a = _info.get("frontal_area_m2", 0.0)
                ss.auto_frontal_area = _front_a
                ss._input_restored = False
                _persist_input_state()        # 새로고침 복구용 디스크 기록

                # 단위셀 도메인은 원점 중심 정사각이라, 형상이 어긋나면 격자가
                # 형상을 못 잡고 Cd=0 이 조용히 나온다. 업로드 즉시 알린다.
                _uv = validate_unit_cell_stl(Path(ss.stl_net_path))
                if not _uv["ok"]:
                    _bb = _uv["bbox"]
                    st.error(
                        "❌ **이 STL 은 단위 셀 모드에 쓸 수 없습니다.**\n\n"
                        + "\n".join(f"- {i}" for i in _uv["issues"])
                        + f"\n\n현재 bbox: x {_bb['x'][0]:.1f}~{_bb['x'][1]:.1f}, "
                          f"y {_bb['y'][0]:.1f}~{_bb['y'][1]:.1f} mm\n\n"
                          "그대로 실행하면 격자가 형상을 잡지 못해 **Cd = 0** 이 "
                          "기록됩니다. 라이노에서 **원점 중심의 정사각 한 칸**으로 "
                          "다시 내보내거나, 전체 구조 모드를 사용하세요.")

                st.info(
                    f"🔍 **STL 자동 감지** — "
                    f"단위 셀 **{_info['cell_size_mm']:.1f} mm**, "
                    f"와이어 직경 **{_info['wire_diameter_mm']:.1f} mm**, "
                    f"고형률 추정 **Sn ≈ {_info['solidity']:.3f}**, "
                    f"실측 기준 투영면적 **{_front_a*1e4:.3f} cm²** (그물면 법선)"
                )

                # STL 정보 표시
                with st.expander("🔍 STL 파일 정보"):
                    st.code(f"""
파일명          : {net_file.name}
크기            : {net_file.size/1024:.1f} KB
저장 경로       : {ss.stl_net_path}
셀 크기(감지)   : {_info['cell_size_mm']:.1f} mm
와이어 직경     : {_info['wire_diameter_mm']:.1f} mm
고형률(추정)    : Sn = {_info['solidity']:.3f}  (참고용 2d/a 근사)
기준 투영면적   : {_front_a:.6e} m²  = {_front_a*1e4:.4f} cm²  (실측 그물면 법선 투영)
                    """)
                    st.caption(
                        "💡 Aref(기준면적)는 학술 표준대로 **고정값**입니다 — 그물면 법선 "
                        "방향 실측 투영면적 × 주기반복수(nx×ny). 영각이 바뀌어도 Aref는 "
                        "고정이며, 영각 효과는 Cd(θ)·Cl(θ) 계수가 표현합니다."
                    )

            elif ss.get("stl_net_path") and Path(ss.stl_net_path).exists():
                # 새로고침 등으로 업로드 위젯은 비었지만 이전 STL이 복구된 경우
                _rn = Path(ss.stl_net_path).name
                st.success(
                    f"✅ 이전 업로드 STL 복구됨: **{_rn}** "
                    f"(미리보기·감지값 유지). 새 파일로 바꾸려면 위에서 다시 업로드하세요."
                )
                if ss.get("auto_cell_size_mm"):
                    _fa = ss.get("auto_frontal_area") or 0.0
                    st.info(
                        f"🔍 **감지값(복구)** — 단위 셀 **{ss.auto_cell_size_mm:.1f} mm**, "
                        f"와이어 직경 **{ss.get('auto_wire_d_mm',0):.1f} mm**, "
                        f"고형률 **Sn ≈ {ss.get('auto_solidity',0):.3f}**, "
                        f"기준 투영면적 **{_fa*1e4:.3f} cm²**"
                    )

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
                _persist_input_state()
            if net_file2:
                ss.stl_net_path = save_uploaded_stl(net_file2, "net2")
                st.success(f"✅ 그물 업로드: {net_file2.name}")
                _persist_input_state()
            # 새로고침 복구 안내 (full_structure)
            if not cage_file and not net_file2 and (
                    ss.get("stl_cage_path") or ss.get("stl_net_path")):
                _parts = []
                if ss.get("stl_cage_path") and Path(ss.stl_cage_path).exists():
                    _parts.append(f"가두리 **{Path(ss.stl_cage_path).name}**")
                if ss.get("stl_net_path") and Path(ss.stl_net_path).exists():
                    _parts.append(f"그물 **{Path(ss.stl_net_path).name}**")
                if _parts:
                    st.success("✅ 이전 업로드 STL 복구됨: " + ", ".join(_parts))

        st.divider()

        # ─── 기준면적 Aref ────────────────────────────────────────────────
        # Cd = Fx / (0.5·ρ·U²·Aref), Cl = Fz / (0.5·ρ·U²·Aref) 의 분모.
        # 자동 계산은 '그물면 법선 투영 + 닫힌 표면 가정(÷2)'이라 그물에는 맞지만
        # 카이트·돛 같은 열린 곡면에는 맞지 않는다 → 직접 입력 경로를 제공한다.
        st.markdown("### 📐 기준면적 (Aref)")
        _ai = _auto_aref_info(mode)

        st.radio(
            "결정 방식", ["자동", "직접 입력"], key="aref_mode", horizontal=True,
            help="Cd·Cl 의 분모가 되는 기준면적입니다. 자동은 STL 형상에서 계산하고, "
                 "직접 입력은 여기 적은 값을 그대로 씁니다.",
        )

        if _ai["value"] is not None:
            st.caption(f"자동 계산값: **{_ai['value']:.6e} m²** "
                       f"({_ai['value']*1e4:.4f} cm²) — {_ai['basis']}")
        else:
            st.caption(f"자동 계산값: 없음 — {_ai['basis']}")

        # 열린 곡면 경고: 자동값이 실제 면적의 절반이 되므로 직접 입력을 권한다.
        if _ai["open_shell"] and ss.aref_mode == "자동":
            st.warning(
                f"⚠️ 이 STL 은 **열린 곡면**(경계 에지 있음)입니다. 자동 계산은 닫힌 "
                f"표면을 가정해 ÷2 하므로 **실제 면적의 절반**이 들어갑니다.\n\n"
                f"- 면 자체 면적(표면적): **{_ai['surface']:.6e} m²**\n"
                f"- 평면 투영면적: **{_ai['projected']:.6e} m²**\n"
                f"- 지금 자동으로 들어갈 값: **{_ai['value']:.6e} m²** ← 절반\n\n"
                f"'직접 입력'으로 바꿔 올바른 값을 지정하세요."
            )

        # 참고값 채우기 버튼이 예약한 값을 위젯 생성 '전에' 반영한다.
        # (위젯 인스턴스화 뒤에 ss[key] 를 쓰면 StreamlitAPIException 이 난다)
        if ss.get("_aref_pending") is not None:
            ss.aref_manual_m2 = float(ss.pop("_aref_pending"))

        # 입력 UI(왼쪽) 옆에 시스템이 계산한 참고값(오른쪽)을 나란히 표시한다.
        _in_col, _ref_col = st.columns([1.15, 1], gap="medium")

        with _in_col:
            if ss.aref_mode == "직접 입력":
                # 처음 전환 시 자동값(열린 곡면이면 표면적)으로 씨앗값을 채워준다.
                if not ss.get("aref_manual_m2"):
                    _seed = (_ai["surface"] if _ai["open_shell"] else _ai["value"]) or 0.0
                    ss.aref_manual_m2 = float(_seed)
                st.number_input(
                    "Aref [m²]", key="aref_manual_m2",
                    min_value=0.0, step=1e-6, format="%.6e",
                    help="0 보다 크면 이 값이 controlDict 의 Aref 로 그대로 들어갑니다. "
                         "0 이면 자동 계산으로 되돌아갑니다.",
                )
                _mv = float(ss.get("aref_manual_m2") or 0.0)
                if _mv > 0:
                    _cmp = ""
                    if _ai["value"]:
                        _cmp = f"  |  자동값 대비 **{_mv/_ai['value']:.3f}배**"
                    st.success(f"✅ 해석에 쓸 Aref = **{_mv:.6e} m²** "
                               f"({_mv*1e4:.4f} cm²){_cmp}")
                else:
                    st.info("Aref 가 0 이라 자동 계산값을 사용합니다.")
                _persist_input_state()
            else:
                st.metric("해석에 쓸 Aref (자동)",
                          f"{_ai['value']*1e4:.3f} cm²" if _ai["value"] is not None else "—")
                st.caption("값을 직접 지정하려면 위에서 '직접 입력'을 선택하세요.")

        with _ref_col:
            # 시스템이 STL 에서 직접 계산한 참고값 — 어떤 모드에서도 항상 보인다.
            st.markdown("**📊 시스템 계산 참고값**")
            if _ai["surface"] is not None:
                st.metric("표면적 (면 자체)", f"{_ai['surface']*1e4:.3f} cm²",
                          help=f"{_ai['surface']:.6e} m² — STL 삼각형 면적의 단순 합. "
                               f"카이트·돛처럼 면 자체 면적을 기준으로 쓸 때의 값입니다.")
                st.metric("투영면적 (실루엣)", f"{_ai['projected']*1e4:.3f} cm²",
                          help=f"{_ai['projected']:.6e} m² — 그물면 법선(Z축) 방향 "
                               f"정면 투영 면적. 그물의 그물발 투영면적이 이 값입니다.")
                if ss.aref_mode == "직접 입력":
                    _b1, _b2 = st.columns(2)
                    if _b1.button("표면적 넣기", use_container_width=True,
                                  key="aref_fill_surface"):
                        ss["_aref_pending"] = _ai["surface"]; st.rerun()
                    if _b2.button("투영면적 넣기", use_container_width=True,
                                  key="aref_fill_proj"):
                        ss["_aref_pending"] = _ai["projected"]; st.rerun()
            else:
                st.caption("STL 을 업로드하면 표면적·투영면적이 계산됩니다.")

        st.divider()

        # ─── Solver 선택 (지시서 §4) ──────────────────────────────────────
        # 기본값은 반드시 Steady. 아무것도 바꾸지 않고 실행하면 기존과 완전히
        # 동일하게 simpleFoam 이 돈다.
        st.markdown("### 🧮 Solver")
        st.radio(
            "해석 방식", ["Steady (simpleFoam)", "Transient (pimpleFoam)"],
            key="solver_mode", horizontal=True,
            help="Steady = 정상상태(기존 동작). Transient = 비정상 해석 후 시간평균. "
                 "구·원기둥처럼 후류가 비정상인 형상의 검증용입니다.",
        )
        _is_tr = ss.solver_mode.startswith("Transient")

        if _is_tr:
            st.warning(
                "⚠️ 비정상 해석은 정상 해석보다 **수십 배** 오래 걸립니다. "
                "지시서 §21에 따라 **단일 검증 케이스 전용**입니다 — "
                "유속·영각을 여러 개로 두지 마세요."
            )
            # Streamlit 은 '렌더되지 않은 위젯'의 key 를 세션에서 제거한다. 이 패널은
            # Solver=Transient 일 때만 렌더되므로, 위젯 key 를 그대로 저장소로 쓰면
            # Steady 로 갔다 오는 순간 값이 사라지고 min_value 로 초기화된다.
            # → 위젯 key(_w_*)와 저장 key(tr_*)를 분리하고 결과를 되써 넣는다.
            def _num(label, skey, **kw):
                ss[skey] = st.number_input(label, value=ss.get(skey), key=f"_w_{skey}", **kw)

            def _chk(label, skey, **kw):
                ss[skey] = st.checkbox(label, value=bool(ss.get(skey)),
                                       key=f"_w_{skey}", **kw)

            _t1, _t2 = st.columns(2)
            with _t1:
                st.markdown("**⏱️ 시간 제어**")
                _num("종료 물리시간 [s]", "tr_end_time",
                     min_value=0.001, step=1.0, format="%.3f")
                _num("초기 시간간격 deltaT [s]", "tr_delta_t",
                     min_value=1e-9, step=1e-4, format="%.6f")
                _num("최대 Courant 수 (maxCo)", "tr_max_co",
                     min_value=0.05, max_value=5.0, step=0.1,
                     help="nOuterCorrectors=1 이면 PIMPLE 이 PISO 로 축약돼 "
                          "Co<1 이 필요합니다. 0.8 권장(보완⑤).")
                _num("최대 시간간격 maxDeltaT [s]", "tr_max_delta_t",
                     min_value=1e-6, step=0.001, format="%.4f")
                _num("결과 저장 간격 [s]", "tr_write_interval",
                     min_value=0.001, step=0.1, format="%.3f")
            with _t2:
                st.markdown("**🔁 PIMPLE 제어**")
                _num("nOuterCorrectors", "tr_n_outer", min_value=1, max_value=20, step=1)
                _num("nCorrectors", "tr_n_corr", min_value=1, max_value=10, step=1)
                _num("nNonOrthogonalCorrectors", "tr_n_non_orth",
                     min_value=0, max_value=10, step=1)
                st.markdown("**🌀 난류 모델**")
                _tm = list(TRANSIENT_TURBULENCE_MODELS.keys())
                ss["tr_turbulence"] = st.selectbox(
                    "모델", _tm,
                    index=_tm.index(ss.get("tr_turbulence", "kOmegaSST"))
                    if ss.get("tr_turbulence") in _tm else 0,
                    key="_w_tr_turbulence",
                    help="기본 kOmegaSST(URANS)는 구·원기둥의 와류 방출을 억제해 "
                         "정상해와 거의 같은 값을 낼 수 있습니다(보완②). 그럴 때 "
                         "DDES 로 바꿔 재검증하세요.")
                if TRANSIENT_TURBULENCE_MODELS.get(ss.tr_turbulence) == "LES":
                    st.caption("⚠️ DES/LES 는 URANS 보다 격자 요건이 훨씬 엄격합니다 "
                               "(후류 등방 정밀화 필요).")

            st.markdown("**▶️ 초기화 · 평균 구간**")
            _t3, _t4 = st.columns(2)
            with _t3:
                _chk("simpleFoam 수렴해를 초기조건으로 사용", "tr_init_steady",
                     help="초기 과도구간을 줄입니다. 다만 구처럼 완전 대칭인 "
                          "형상은 대칭해에서 출발하면 와류가 안 생길 수 있습니다(보완③).")
                if ss.tr_init_steady:
                    _num("선행 simpleFoam 반복 횟수", "tr_steady_iters",
                         min_value=100, max_value=20000, step=100)
                else:
                    _chk("대칭 교란 주입 (보완③)", "tr_perturb",
                         help="초기 속도장에 자유류의 일정 비율만큼 횡방향 성분을 "
                              "더해 대칭을 깹니다. 구·원기둥에 권장.")
                    if ss.tr_perturb:
                        _num("교란 크기 (자유류 대비 비율)", "tr_perturb_mag",
                             min_value=0.0001, max_value=0.2, step=0.005, format="%.4f")
            with _t4:
                _num("평균 시작 시각 TavgStart [s]  (0 = 자동)", "tr_avg_start",
                     min_value=0.0, step=1.0,
                     help="이 시각 이후 데이터만 시간평균합니다. 0 이면 "
                          "전체 구간의 50% 지점을 자동 사용합니다.")
                _L = float(ss.get("cell_size_mm", 20.0)) / 1000.0
                _U = max(float(ss.get("u_min", 1.0)), 1e-6)
                st.caption(f"권장 TavgStart ≳ 10·L/U ≈ {10*_L/_U:.2f} s "
                           f"(L=대표길이, U=유속 기준)")

        st.divider()

        # ─── 계산량 프리셋 ────────────────────────────────────────────────
        st.markdown("### ⚡ 계산량 프리셋")
        # desc의 예상시간은 estimate_case_minutes(실측 재보정)와 일치하도록 갱신.
        _PRESETS = {
            "최소":     {"end_time": 500,   "refine_level": 2, "residual": "1e-3", "write_interval": 100,
                         "desc": "빠른 테스트 (~1~2분)"},
            "보통":     {"end_time": 2000,  "refine_level": 3, "residual": "1e-4", "write_interval": 100,
                         "desc": "일반 해석 (~5분)"},
            "정밀":     {"end_time": 5000,  "refine_level": 4, "residual": "1e-4", "write_interval": 200,
                         "desc": "고정밀 (~25분)"},
            "최고정밀": {"end_time": 10000, "refine_level": 4, "residual": "1e-5", "write_interval": 500,
                         "desc": "검증용 (~45분)"},
        }
        # v12 항목1: 프리셋 적용은 on_click 콜백으로 — 종전의 본문 내
        # st.rerun() 은 스크립트를 조기 중단시켜 그 아래에서 생성되는 위젯
        # (유속/영각 단계 수 등)의 상태를 Streamlit 이 청소 → 기본값(1)으로
        # 리셋되는 버그를 유발했다. 콜백은 다음 런 '시작 전'에 실행되므로
        # 전체 위젯이 정상 렌더되고 추정시간·해석조건이 즉시 반영된다.
        def _apply_calc_preset(_pname: str):
            _pv = _PRESETS[_pname]
            ss.calc_preset_name      = _pname
            ss.end_time_preset       = _pv["end_time"]
            ss.refine_level_preset   = _pv["refine_level"]
            ss.residual_preset       = _pv["residual"]
            ss.write_interval_preset = _pv["write_interval"]

        _pcols = st.columns(4)
        for _col, (_pname, _pvals) in zip(_pcols, _PRESETS.items()):
            with _col:
                _is_active = (ss.get("calc_preset_name") == _pname)
                _btn_type = "primary" if _is_active else "secondary"
                st.button(
                    f"{'✅ ' if _is_active else ''}{_pname}",
                    help=_pvals["desc"],
                    use_container_width=True,
                    type=_btn_type,
                    key=f"preset_btn_{_pname}",
                    on_click=_apply_calc_preset, args=(_pname,),
                )

        _cur = _PRESETS.get(ss.get("calc_preset_name", "보통"), _PRESETS["보통"])
        st.caption(
            f"현재: **{ss.get('calc_preset_name','보통')}** — "
            f"반복 {ss.get('end_time_preset', 2000)} / 격자 레벨 {ss.get('refine_level_preset', 3)} / "
            f"수렴 {ss.get('residual_preset','1e-4')} / 저장간격 {ss.get('write_interval_preset', 100)}"
        )

        # 수렴 기준 · 저장 간격 (개별 조정 가능)
        _sc1, _sc2 = st.columns(2)
        with _sc1:
            residual_control = st.select_slider(
                "수렴 기준 (Residual)",
                options=["1e-3", "1e-4", "1e-5"],
                key="residual_preset",
                help="프리셋으로 자동 설정됩니다. 수동 조정 가능.",
            )
            residual_control = float(residual_control)
        with _sc2:
            write_interval = st.number_input(
                "결과 저장 간격 (스텝)",
                min_value=10, max_value=1000, step=50,
                key="write_interval_preset",
                help="N 스텝마다 결과 필드를 디스크에 저장합니다.",
            )

        st.divider()

        # ─── 해석 파라미터 ────────────────────────────────────────────────
        st.markdown("### 🎛️ 해석 파라미터")

        # ─── STL 유형 판별 및 권장 설정 (요구서 §3·§4·§13) ────────────────
        # 형상마다 적절한 설정이 다르므로, 먼저 무엇인지 판별한 뒤 권장값을 낸다.
        # 자동 분류가 확실하지 않으면 임의로 고르지 않고 사용자에게 확인받는다.
        _stl_for_type = ss.get("stl_net_path") or ss.get("stl_cage_path")
        if _stl_for_type and Path(_stl_for_type).exists():
            with st.expander("🔎 STL 유형 판별 · 권장 설정", expanded=False):
                if ss.get("_cls_path") != _stl_for_type:
                    ss["_cls"] = classify_stl(Path(_stl_for_type))
                    ss["_cls_path"] = _stl_for_type
                    ss["stl_type_user"] = "자동 판별 결과 사용"
                _cls = ss.get("_cls") or {}
                _feat = _cls.get("features", {})

                if _cls.get("needs_confirm"):
                    st.warning(
                        f"STL 형상을 자동으로 분류하기 어렵습니다"
                        f"(가장 가까운 후보: **{STL_TYPE_LABELS.get(_cls.get('type_guess',''), '—')}**, "
                        f"신뢰도 {_cls.get('confidence',0)*100:.0f}%). 아래에서 직접 선택하십시오.")
                else:
                    st.success(f"자동 판별: **{_cls.get('label','—')}** "
                               f"(신뢰도 {_cls.get('confidence',0)*100:.0f}%)")
                why(_cls.get("basis", "") + "\n\n측정값 — " + (
                    f"삼각형 {_feat.get('n_tri',0):,}개 · "
                    f"바운딩박스 {' × '.join(f'{s:.1f}' for s in _feat.get('spans',[0,0,0]))} mm · "
                    f"표면적 {_feat.get('area_mm2',0):,.0f} mm² · "
                    f"닫힌 표면 {'예' if _feat.get('closed') else '아니오'}"))

                _opts = ["자동 판별 결과 사용"] + [STL_TYPE_LABELS[k] for k in
                                                 ("sphere", "cylinder", "kite",
                                                  "net_panel", "complex")]
                ss["stl_type_user"] = st.radio(
                    "형상 유형", _opts, index=_opts.index(ss.get("stl_type_user", _opts[0]))
                    if ss.get("stl_type_user") in _opts else 0,
                    key="_w_stl_type_user",
                    help="자동 판별이 틀렸다고 판단되면 직접 지정하십시오. "
                         "선택한 유형에 맞춰 아래 권장 설정이 바뀝니다.")

                _rev = {v: k for k, v in STL_TYPE_LABELS.items()}
                _eff_type = (_cls.get("type") if ss["stl_type_user"] == _opts[0]
                             else _rev.get(ss["stl_type_user"], "unknown"))
                if _eff_type in ("unknown", None):
                    st.info("유형이 정해지지 않아 권장 설정을 제시하지 않습니다. "
                            "위에서 형상을 선택하십시오.")
                else:
                    _pre = STL_TYPE_PRESETS[_eff_type]
                    st.markdown(f"**{STL_TYPE_LABELS[_eff_type]} 권장 설정**")
                    _rows = [
                        {"항목": "해석 모드", "권장": ("단위 셀" if _pre["analysis_mode"] == "unit_cell"
                                                  else "전체 구조")},
                        {"항목": "Solver", "권장": _pre["solver"]},
                        {"항목": "정밀화 레벨", "권장": str(_pre["refine_level"])},
                        {"항목": "격자 자동 보정", "권장": "켬" if _pre["auto_refine"] else "끔"},
                        {"항목": "기준면적", "권장": _pre["aref_mode"]},
                    ]
                    if _pre.get("net_grid_redesign"):
                        _rows.append({"항목": "배경격자 재설계",
                                      "권장": f"켬 · 목표 {_pre['net_grid_target_cells']:.0f} 셀"})
                    st.dataframe(_rows, use_container_width=True, hide_index=True)
                    why(f"Solver 권장 근거 — {_pre['solver_reason']}", "왜? (Solver)")
                    for _w in _pre.get("warnings", []):
                        st.warning(_w)

                    if ss.pop("_preset_applied_msg", None):
                        st.success("권장 설정을 적용했습니다. 값은 언제든 직접 바꿀 수 "
                                   "있습니다(권장일 뿐 강제가 아닙니다).")
                    if st.button("권장 설정 적용", key="_apply_preset"):
                        _pp = {
                            "analysis_mode":      _pre["analysis_mode"],
                            "solver_mode":        ("Transient (pimpleFoam)"
                                                   if _pre["solver"] == "Transient"
                                                   else "Steady (simpleFoam)"),
                            "refine_level_preset": int(_pre["refine_level"]),
                            "auto_refine":        bool(_pre["auto_refine"]),
                            "aref_mode":          _pre["aref_mode"],
                        }
                        if _pre.get("net_grid_redesign"):
                            _pp["net_grid_redesign"] = True
                            _pp["net_grid_target_cells"] = float(_pre["net_grid_target_cells"])
                        # 위젯 생성 전 단계에서 반영해야 하므로 예약만 하고 rerun
                        ss["_pending_preset"] = _pp
                        st.rerun()

        # 격자 자동 보정 — 두 모드 공통. 배경격자는 형상 전체 크기 기준으로
        # 정해지므로, 그물처럼 큰 영역에 가는 요소가 흩어진 형상은 기본 레벨에서
        # 실 지름당 1~2 셀밖에 안 걸린다(3by3 실측: 레벨3 1.5셀 → Cd 43% 과대).
        st.checkbox(
            "🔧 형상 최소두께 기준으로 정밀화 레벨 자동 보정 (권장)",
            key="auto_refine",
            help="STL 의 가장 가는 치수(그물실 지름·판재 두께)를 읽어 셀이 "
                 "10개 이상 걸리도록 정밀화 레벨을 자동으로 올립니다. "
                 "STL 종류와 무관하게 동작하며, 굵은 형상(구·카이트)처럼 이미 "
                 "충분하면 레벨을 올리지 않습니다. 계산시간이 크게 늘 수 있습니다.")
        if ss.get("auto_refine"):
            st.caption("적용 예 — 그물 3by3(실 3mm): 레벨 3 → **6** 자동 상향 · "
                       "카이트(두께 78mm)·구(300mm): 레벨 3 유지")
        else:
            st.caption("⚠️ 꺼져 있습니다. 가는 그물실은 격자가 부족해 Cd 가 "
                       "과대평가될 수 있습니다(로그에 경고가 남습니다).")

        # ─── 전체구조 격자 설계 (배경격자 재설계 · 후류 박스 레벨) ──────────
        # 조건부로 그려지는 위젯은 key 를 그대로 저장소로 쓰면 패널이 닫힐 때
        # 값이 사라지므로, 위젯 key(_w_*)와 저장 key 를 분리한다(v18 과 동일).
        if mode == "full_structure":
            with st.expander("🕸️ 그물 격자 설계 (전체구조)", expanded=False):
                ss["net_grid_redesign"] = st.checkbox(
                    "배경격자 재설계 — 임계 치수 기준으로 배경 셀을 정한다",
                    value=bool(ss.get("net_grid_redesign")),
                    key="_w_net_grid_redesign",
                    help="종전 배경격자는 '형상 전체 크기 L/8' 기준이라 그물처럼 "
                         "가는 요소가 흩어진 형상은 실 지름당 1~2 셀에 그칩니다. "
                         "재설계는 base = 임계치수 × 2^레벨 / 목표 로 잡아 지정한 "
                         "레벨에서 곧바로 목표 해상도가 나오게 합니다. 도메인도 "
                         "상류 2L·하류 5L·횡 ±2L 로 축소됩니다(부피 1/2.7).")
                if ss["net_grid_redesign"]:
                    ss["net_grid_target_cells"] = st.number_input(
                        "목표 셀 수 / 임계치수(그물실 지름)",
                        value=float(ss.get("net_grid_target_cells", 75.0)),
                        min_value=10.0, max_value=300.0, step=5.0,
                        key="_w_net_grid_target",
                        help="실측: 40셀 Cd=0.776 / 75셀 0.936 (같은 격자 DDES 0.993). "
                             "값이 클수록 셀 수와 계산시간이 급격히 늘어납니다.")
                    st.caption("실측 — 목표 75: 196만 셀 · 정상해 74분 / "
                               "목표 150: 647만 셀 · 정상해 280분")
                else:
                    st.caption("꺼져 있으면 종전 배경격자(형상 L/8)를 씁니다.")

                st.radio("후류 정밀화 박스 레벨", ["자동", "직접 지정"],
                         key="wake_box_mode", horizontal=True,
                         help="자동은 셀 폭발을 막기 위해 표면 레벨이 4 이상이면 "
                              "박스를 레벨 2 로 묶습니다. DES/LES 는 후류가 임계 "
                              "치수당 5셀 이상이어야 LES 모드로 전환되므로, 위 "
                              "'격자 적정성 판정'에서 후류 부족이 뜨면 여기서 "
                              "권고 레벨로 올리십시오.")
                if ss.get("wake_box_mode") == "직접 지정":
                    ss["wake_box_level"] = int(st.number_input(
                        "후류 박스 레벨", value=int(ss.get("wake_box_level", 2)),
                        min_value=0, max_value=8, step=1, key="_w_wake_box_level",
                        help="박스는 체적을 통째로 세분하므로 레벨을 1 올릴 때마다 "
                             "박스 안 셀이 8배가 됩니다. 표면 레벨을 넘지 않습니다."))
                    st.caption("⚠️ 레벨 1 상승 = 박스 내 셀 8배. 표면 레벨보다 크게 "
                               "잡아도 표면 레벨로 잘립니다.")

        # ─── 임계 최소 치수 · 격자 적정성 ────────────────────────────────
        # 격자가 반드시 해상해야 하는 건 형상 전체 크기가 아니라 '가장 가는 부분'
        # (그물실 지름·판재 두께)이다. 이 값으로 표면과 후류를 따로 판정한다.
        _stl_for_crit = ss.get("stl_net_path")
        if _stl_for_crit and Path(_stl_for_crit).exists():
            with st.expander("📏 임계 최소 치수 · 격자 적정성 판정", expanded=False):
                _cd = critical_dimension(Path(_stl_for_crit))
                _c1, _c2 = st.columns([1, 1.1])
                with _c1:
                    st.radio("임계 치수 결정", ["자동", "직접 입력"],
                             key="crit_dim_mode", horizontal=True,
                             help="격자가 해상해야 할 가장 가는 치수입니다. "
                                  "자동은 STL 바운딩박스의 최소변을 씁니다.")
                    st.caption(f"자동 산정: **{_cd['bbox_min']:.3f} mm** "
                               f"({_cd['basis']})")
                    if _cd.get("thickness"):
                        st.caption(f"참고 — 체적/표면적 기반 추정 6V/A = "
                                   f"{_cd['thickness']:.3f} mm")
                    if ss.crit_dim_mode == "직접 입력":
                        if not ss.get("crit_dim_manual_mm"):
                            ss.crit_dim_manual_mm = float(_cd["bbox_min"] or 1.0)
                        ss.crit_dim_manual_mm = st.number_input(
                            "임계 치수 [mm]", value=float(ss.crit_dim_manual_mm),
                            min_value=0.001, step=0.1, format="%.3f",
                            key="_w_crit_dim")
                _crit = (float(ss.crit_dim_manual_mm)
                         if ss.crit_dim_mode == "직접 입력" and ss.get("crit_dim_manual_mm")
                         else float(_cd["bbox_min"] or 0.0))

                # 배경격자: 단위셀은 a/16, 전체구조는 형상 L/8
                if mode == "unit_cell":
                    # 이 블록은 모드 분기보다 앞서므로 위젯 변수(cell_size) 대신
                    # 세션 값을 쓴다(단위: mm → m).
                    _base = unit_cell_base_mm(
                        float(ss.get("cell_size_mm", 20.0)) / 1000.0)
                else:
                    _spans = _cd.get("spans") or [1.0]
                    _base = max(_spans) / 8.0
                    if ss.get("net_grid_redesign"):
                        # 재설계를 켜면 배경격자 산식이 바뀐다. 빌더와 같은 함수를
                        # 써서 판정이 실제 생성 격자와 어긋나지 않게 한다.
                        _L = max(_spans) / 1000.0          # mm → m (도메인 단위 통일)
                        _base = net_grid_base_cell(
                            _crit, int(ss.get("refine_level_preset", 3)),
                            float(ss.get("net_grid_target_cells", 75.0)),
                            domain_m=(7.0 * _L, 4.0 * _L, 4.0 * _L)) * 1000.0
                _fam = ("LES" if TRANSIENT_TURBULENCE_MODELS.get(
                            ss.get("tr_turbulence", "kOmegaSST")) == "LES"
                        and str(ss.get("solver_mode", "")).startswith("Transient")
                        else "RAS")
                # refine_level 위젯도 이 블록보다 뒤에 생성되므로 세션 값 사용
                _rl = int(ss.get("refine_level_preset", 3))
                # 근접(후류) 정밀화 레벨은 모드마다 규칙이 다르다.
                #   단위셀   : 거리 2mm 이내 영역 = min(3, 레벨)
                #   전체구조 : refineBox = 레벨 3 이하면 레벨-1, 아니면 2
                # 종전에는 두 모드 모두 전체구조 규칙으로 표시해 단위셀 판정이
                # 실제 격자와 어긋났다.
                if mode == "unit_cell":
                    _box_lv = min(3, _rl)
                else:
                    _box_lv = _rl - 1 if _rl <= 3 else 2
                    if ss.get("wake_box_mode") == "직접 지정":
                        _box_lv = min(_rl, int(ss.get("wake_box_level", 2)))
                _ad = mesh_adequacy(_crit, _base, _rl, _box_lv, _fam)

                with _c2:
                    st.markdown(f"**현재 설정 판정** — 난류 {_ad['model_family']}")
                    st.write(f"- 임계 치수 **{_crit:.3f} mm** · 배경격자 {_base:.3f} mm")
                    st.write(f"- 표면 정밀화 lv{_ad['surface_level']} → 셀 "
                             f"{_ad['surface_cell_mm']:.4f} mm = "
                             f"**{_ad['surface_cells']:.1f}셀** "
                             + ("✅" if _ad["surface_ok"] else
                                f"⚠️ (목표 {SURF_CELLS_TARGET:.0f})"))
                    if _ad["needs_wake"]:
                        st.write(f"- 후류 박스 lv{_ad['box_level']} → 셀 "
                                 f"{_ad['wake_cell_mm']:.4f} mm = "
                                 f"**{_ad['wake_cells']:.1f}셀** "
                                 + ("✅" if _ad["wake_ok"] else
                                    f"⚠️ (목표 {WAKE_CELLS_TARGET:.0f})"))
                    if not _ad["ok"]:
                        st.warning(
                            f"⚠️ 권고: 표면 정밀화 **레벨 {_ad['surface_required']} 이상**"
                            + (f", 후류 박스 **레벨 {_ad['wake_required']} 이상**"
                               if _ad["needs_wake"] and not _ad["wake_ok"] else "")
                            + ".\n\n후류가 부족하면 DES/LES 가 LES 모드로 전환하지 "
                              "못해 사실상 RANS 로 동작합니다(실측: 3by3 그물에서 "
                              "DDES 변동 0.02%).")
                    else:
                        st.success("✅ 현재 설정으로 충분합니다.")

                st.markdown("**레벨별 적정성**")
                _rows = []
                for r in mesh_adequacy_table(_crit, _base):
                    _rows.append({
                        "레벨": r["level"],
                        "셀 크기 [mm]": f"{r['cell_mm']:.4f}",
                        "임계치수당 셀": f"{r['cells']:.1f}",
                        f"표면 RANS (≥{SURF_CELLS_TARGET:.0f})": "OK" if r["surface_ok"] else "부족",
                        f"후류 DES/LES (≥{WAKE_CELLS_TARGET:.0f})": "OK" if r["wake_ok"] else "부족",
                    })
                st.dataframe(_rows, use_container_width=True, hide_index=True)

        if mode == "unit_cell":
            # ── 유속·영각 범위 (단계수 1×1 = 단일 해석) ──────────────────
            # 단일/배치 해석을 하나의 인터페이스로 통합한다. 단계수를 모두 1로
            # 두면 1개 케이스(=단일 해석), 2 이상이면 조합 배치 해석.
            st.markdown("#### 🌊 유속·영각 범위  (단계수 1 = 단일 해석)")
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                u_min = st.number_input("최소 유속 [m/s]", value=1.0,
                                        min_value=0.1, max_value=5.0, step=0.1,
                                        key="u_min")
                u_max = st.number_input("최대 유속 [m/s]", value=1.0,
                                        min_value=0.1, max_value=5.0, step=0.1,
                                        key="u_max")
                u_steps = st.number_input("유속 단계 수", value=1,
                                          min_value=1, max_value=20, step=1,
                                          key="u_steps")
            with _rc2:
                a_min = st.number_input("최소 영각 [°]", value=0.0,
                                        min_value=-90.0, max_value=90.0, step=5.0,
                                        key="a_min")
                a_max = st.number_input("최대 영각 [°]", value=0.0,
                                        min_value=-90.0, max_value=90.0, step=5.0,
                                        key="a_max")
                a_steps = st.number_input("영각 단계 수", value=1,
                                          min_value=1, max_value=20, step=1,
                                          key="a_steps")
            speeds = [float(s) for s in np.linspace(u_min, u_max, int(u_steps))]
            angles = [float(a) for a in np.linspace(a_min, a_max, int(a_steps))]
            speed_val, angle_val = speeds[0], angles[0]   # 대표값(미리보기·요약)
            _ncase = len(speeds) * len(angles)
            if _ncase == 1:
                st.caption("단계수 **1 × 1 → 단일 해석**으로 실행됩니다.")
            else:
                st.caption(f"총 **{_ncase}개** 케이스 배치 해석 "
                           f"(유속 {len(speeds)} × 영각 {len(angles)}).")

            # ── 격자·형상 파라미터 ───────────────────────────────────────
            _pc1, _pc2 = st.columns(2)
            with _pc1:
                _auto_cs = ss.get("auto_cell_size_mm")
                _cs_help = ("STL 자동 감지값이 적용됨. 수동 수정 가능."
                            if _auto_cs else "STL 업로드 시 자동 감지됩니다.")
                # value= 생략 (session_state 규칙 — init_session 시드/STL 갱신값 사용)
                cell_size = st.number_input(
                    "단위 셀 크기 a [mm]",
                    min_value=1.0, max_value=200.0, step=1.0,
                    key="cell_size_mm",
                    help=_cs_help,
                ) / 1000.0
                if _auto_cs:
                    st.caption(f"자동 감지: {_auto_cs:.1f} mm | 와이어: {ss.get('auto_wire_d_mm',0):.1f} mm")
                solidity = st.number_input(
                    "고형률 Sn (그물실 투영 면적 비율)",
                    min_value=0.01, max_value=0.95, step=0.01,
                    key="solidity_input",
                    help="Sn = 그물실 투영 면적 / 패널 전체 면적. STL 업로드 시 자동 추정 (≈ 2d/a).",
                )
            with _pc2:
                refine_level = st.number_input(
                    "격자 정밀화 레벨",
                    min_value=1, max_value=8, step=1,
                    key="refine_level_preset",
                    help="snappyHexMesh 표면 최대 정밀화 레벨 (min = 레벨-1). "
                         "레벨 3: ~50만 셀(권장), 레벨 4: ~200만 셀(정밀). "
                         "그물실이 가늘면 레벨을 더 올려야 합니다(아래 경고 참조).",
                )
                # ── 그물실 해상도 경고 ──────────────────────────────────
                # 배경격자가 망목 크기에 비례해 정해지므로, 망목이 커지면 실 대비
                # 격자가 사용자 모르게 거칠어진다. 항상 숫자로 보여주고 부족하면
                # 필요한 레벨을 알려준다.
                _wd = float(ss.get("auto_wire_d_mm") or 0.0)
                if _wd > 0:
                    _trw = twine_resolution(unit_cell_base_mm(cell_size), _wd,
                                            int(refine_level))
                    if _trw["ok"]:
                        st.success(
                            f"✅ 그물실 해상도 **{_trw['cells_per_d']:.1f} 셀/지름** "
                            f"(목표 {TWINE_CELLS_TARGET:.0f} 이상)")
                    else:
                        st.warning(
                            f"⚠️ 그물실 해상도 **{_trw['cells_per_d']:.1f} 셀/지름** "  # noqa: E501
                            f"— 목표 {TWINE_CELLS_TARGET:.0f} 셀 미만입니다.\n\n"
                            f"실 지름 {_wd:.2f} mm 에 최소 셀이 "
                            f"{_trw['finest_mm']:.3f} mm 라 원통 표면의 경계층·박리를 "
                            f"풀지 못합니다. Cd 가 부정확할 수 있습니다.\n\n"
                            f"**정밀화 레벨을 {_trw['required_level']} 이상**으로 "
                            f"올리세요 (레벨 1 상승마다 표면 근처 셀이 약 8배).")

                end_time = st.number_input(
                    "최대 반복 횟수",
                    min_value=100, max_value=10000, step=100,
                    key="end_time_preset",
                    help="controlDict endTime. 수렴 기준 도달 시 조기 종료됩니다.",
                )

            # 대표 조건(목록 첫 값) 속도 벡터 미리보기
            from cfd_manager import compute_velocity_vector
            Ux, Uy, Uz = compute_velocity_vector(speed_val, angle_val)
            st.info(
                f"📐 **대표 속도 벡터** (U={speed_val:.2f} m/s, α={angle_val:.1f}°) "
                f"= ({Ux:.3f}, 0, {Uz:.3f}) m/s  "
                f"| Reynolds = {speed_val * _re_length_m('unit_cell')[0] / _nu_si():.1f}"
            )
            _L, _Lb = _re_length_m("unit_cell")
            st.caption(
                f"Reynolds = ρUL/μ = U·L/ν · 대표 길이 L = **{_fmt_len(_L)}** "
                f"({_Lb}) · ν = **{_nu_si():.4g} m²/s** (물리 조건 입력값)")
            st.radio("대표 길이 기준", ["자동", "바운딩박스 최대변", "임계 최소 치수", "직접 입력"],
                     key="re_length_mode", horizontal=True,
                     help="Reynolds 수의 대표 길이를 무엇으로 볼지 정합니다. "
                          "형상에 따라 관례가 다르므로 프로그램이 하나로 강제하지 않습니다.")
            if ss.get("re_length_mode") == "직접 입력":
                ss["re_length_manual_mm"] = st.number_input(
                    "대표 길이 [mm]", value=float(ss.get("re_length_manual_mm") or cell_size*1000),
                    min_value=0.001, step=1.0, key="_w_re_len_uc")

            # 주기 경계조건 반복 수 (보고용 — 메시/Cd/계산시간에 영향 없음)
            st.markdown("#### 🔁 주기 반복 수 (Nx×Ny) — 보고용")
            col_nx, col_ny = st.columns(2)
            with col_nx:
                nx_val = st.number_input(
                    "X 방향 반복 수 (Nx)",
                    min_value=1, max_value=100, step=1,
                    key="unit_nx",
                    help="대표하려는 무한 배열의 X 방향 셀 개수. 격자는 항상 1셀만 "
                         "계산하므로 Cd·계산시간에는 영향 없음. '총 힘[N]' 환산에만 사용."
                )
            with col_ny:
                ny_val = st.number_input(
                    "Y 방향 반복 수 (Ny)",
                    min_value=1, max_value=100, step=1,
                    key="unit_ny",
                    help="대표하려는 무한 배열의 Y 방향 셀 개수. (Cd·계산시간 불변)"
                )
            domain_x_mm = cell_size * 1000
            domain_y_mm = cell_size * 1000
            st.info(
                f"📦 **해석 격자**: 항상 **1셀** = {domain_x_mm:.1f} × {domain_y_mm:.1f} mm "
                f"(주기 BC). Nx×Ny={nx_val}×{ny_val}는 **계산시간·Cd에 영향 없음** — "
                f"총 힘[N]을 {nx_val*ny_val}배로 환산할 때만 쓰입니다."
            )

        else:  # full_structure
            # ── 유속·영각 범위 (단계수 1×1 = 단일 해석) — 단위셀과 키 공유 ──
            st.markdown("#### 🌊 유속·영각 범위  (단계수 1 = 단일 해석)")
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                u_min = st.number_input("최소 유속 [m/s]", value=1.0,
                                        min_value=0.1, max_value=5.0, step=0.1,
                                        key="u_min")
                u_max = st.number_input("최대 유속 [m/s]", value=1.0,
                                        min_value=0.1, max_value=5.0, step=0.1,
                                        key="u_max")
                u_steps = st.number_input("유속 단계 수", value=1,
                                          min_value=1, max_value=20, step=1,
                                          key="u_steps")
            with _rc2:
                a_min = st.number_input("최소 영각 [°]", value=0.0,
                                        min_value=-90.0, max_value=90.0, step=5.0,
                                        key="a_min")
                a_max = st.number_input("최대 영각 [°]", value=0.0,
                                        min_value=-90.0, max_value=90.0, step=5.0,
                                        key="a_max")
                a_steps = st.number_input("영각 단계 수", value=1,
                                          min_value=1, max_value=20, step=1,
                                          key="a_steps")
            speeds = [float(s) for s in np.linspace(u_min, u_max, int(u_steps))]
            angles = [float(a) for a in np.linspace(a_min, a_max, int(a_steps))]
            speed_val, angle_val = speeds[0], angles[0]
            _ncase = len(speeds) * len(angles)
            st.caption("단계수 **1 × 1 → 단일 해석**으로 실행됩니다." if _ncase == 1
                       else f"총 **{_ncase}개** 케이스 배치 해석.")

            _gc1, _gc2 = st.columns(2)
            with _gc1:
                cage_d = st.number_input("가두리 직경 D [m]", value=10.0,
                                         min_value=1.0, max_value=50.0, step=0.5,
                                         key="cage_d")
            with _gc2:
                cage_h = st.number_input("가두리 수심 H [m]", value=5.0,
                                         min_value=1.0, max_value=30.0, step=0.5,
                                         key="cage_h")
            end_time = st.number_input(
                "최대 반복 횟수",
                min_value=100, max_value=15000, step=100,
                key="end_time_preset",
                help="controlDict endTime. 수렴 기준 도달 시 조기 종료됩니다.",
            )
            _L, _Lb = _re_length_m("full_structure")
            Re = speed_val * _L / _nu_si()
            st.info(f"📐 **Reynolds 수** = {Re:.2e}  |  도메인: {3*cage_d:.0f}D × {3*cage_d:.0f}D × {cage_h:.0f}m")
            st.caption(
                f"Reynolds = ρUL/μ = U·L/ν · 대표 길이 L = **{_fmt_len(_L)}** "
                f"({_Lb}) · ν = **{_nu_si():.4g} m²/s** (물리 조건 입력값)")
            st.radio("대표 길이 기준", ["자동", "바운딩박스 최대변", "임계 최소 치수", "직접 입력"],
                     key="re_length_mode", horizontal=True,
                     help="Reynolds 수의 대표 길이를 무엇으로 볼지 정합니다. "
                          "형상에 따라 관례가 다르므로 프로그램이 하나로 강제하지 않습니다.")
            if ss.get("re_length_mode") == "직접 입력":
                ss["re_length_manual_mm"] = st.number_input(
                    "대표 길이 [mm]", value=float(ss.get("re_length_manual_mm") or cage_d*1000),
                    min_value=0.001, step=1.0, key="_w_re_len_fs")

    with col_right:
        # ─── STL 미리보기 (인터랙티브 3D) ────────────────────────────────
        _stl_show = ss.get("stl_net_path") or ss.get("stl_cage_path")
        if _stl_show and Path(_stl_show).exists():
            st.markdown("### 🖼️ STL 미리보기")
            _angle_v  = float(angle_val)   # 대표 영각(목록 첫 값)
            _nx_v     = int(ss.get("unit_nx", 1))
            _ny_v     = int(ss.get("unit_ny", 1))
            _cs_m     = float(ss.get("cell_size_mm", 20.0)) / 1000.0

            # 표면적 집계 영역을 녹색으로 확인하는 토글(기본 꺼짐 — 켤 때만 표시)
            st.toggle(
                "🟩 표면적 영역 표시", key="show_surface_area",
                help="시스템이 계산한 표면적에 실제로 포함된 면을 녹색으로 칠합니다. "
                     "Nx×Ny 타일링 시에는 면적 집계 대상인 기준 타일 1개만 녹색이 됩니다.",
            )

            # ── 인터랙티브 Plotly 뷰어 (마우스 드래그 회전 가능) ──────────
            # 첫 렌더에만 카메라를 지정하고, 이후 영각·nx·ny 변경 렌더에서는 빼서
            # uirevision이 사용자의 마우스 카메라를 보존하게 한다.
            _stl_init_cam = not ss.get("_cam_init_stl", False)
            _fig3d = render_stl_interactive_plotly(
                _stl_show, mode, _angle_v, _nx_v, _ny_v, _cs_m,
                init_camera=_stl_init_cam,
                highlight_surface=bool(ss.get("show_surface_area", False)),
            )
            if _fig3d is not None:
                st.plotly_chart(_fig3d, use_container_width=True,
                                key="stl_3d_preview")
                ss["_cam_init_stl"] = True
                if ss.get("show_surface_area"):
                    # 녹색 영역이 곧 표면적 수치임을 숫자로 함께 확인시켜 준다.
                    _si = _auto_aref_info(mode)
                    if _si["surface"] is not None:
                        st.caption(
                            f"🟩 녹색 = 표면적 집계 영역  |  "
                            f"**{_si['surface']:.6e} m²** ({_si['surface']*1e4:.3f} cm²)"
                            + ("  ·  파란색 복제 타일은 집계 제외"
                               if (mode == "unit_cell" and (_nx_v > 1 or _ny_v > 1)) else "")
                        )
                st.caption(
                    "💡 마우스 드래그: 회전  |  스크롤: 줌  |  오른쪽 드래그: 이동"
                )
            else:
                # ── 정적 이미지 폴백 ──────────────────────────────────────
                _stl_img = render_stl_preview(_stl_show, angle_deg=_angle_v)
                _mode_badge = "🔬 단위 셀 모드" if mode == "unit_cell" else "🏗️ 전체 구조 모드"
                if _stl_img:
                    st.image(_stl_img, use_container_width=True,
                             caption=f"{_mode_badge}  |  유속 방향 α={_angle_v:.0f}°")
                else:
                    st.info("STL 렌더링 실패 — Plotly/PyVista/matplotlib 확인 필요")

            if mode == "unit_cell" and ss.get("auto_cell_size_mm"):
                _ci1, _ci2, _ci3, _ci4 = st.columns(4)
                _ci1.metric("셀 크기", f"{ss.auto_cell_size_mm:.1f} mm")
                _ci2.metric("와이어 직경", f"{ss.get('auto_wire_d_mm',0):.1f} mm")
                _ci3.metric("고형률 Sn", f"{ss.get('auto_solidity',0):.3f}")
                # 투영면적 — 그물면 법선 투영(실측) × 반복수(Nx×Ny). 표시는 총면적,
                # 해석용 Aref는 1셀 고정(Cd 불변)임에 유의(요약 주석 참조).
                try:
                    from cfd_manager import compute_projected_area as _cpa
                    _aref_cell = _cpa(Path(ss.stl_net_path), (0.0, 0.0, 1.0))
                    _nxny = int(ss.get("unit_nx", 1)) * int(ss.get("unit_ny", 1))
                    _ci4.metric("투영면적 (총)", f"{_aref_cell*_nxny*1e4:.3f} cm²",
                                help=f"그물면 법선 투영(실측) × 반복수 {_nxny} "
                                     f"(1셀 {_aref_cell*1e4:.3f} cm²). 해석용 Aref는 "
                                     f"1셀 고정이라 Cd는 nx/ny에 불변.")
                except Exception:
                    pass
            st.divider()

        # ─── 해석 정보 요약 ───────────────────────────────────────────────
        st.markdown("### 📋 해석 설정 요약")

        _spd_txt = (f"{speeds[0]:.2f} m/s" if len(speeds) == 1
                    else f"{speeds[0]:.2f}~{speeds[-1]:.2f} m/s ({len(speeds)}단계)")
        _ang_txt = (f"{angles[0]:.1f}°" if len(angles) == 1
                    else f"{angles[0]:.1f}~{angles[-1]:.1f}° ({len(angles)}단계)")
        summary_data = {
            "해석 모드":    "단위 셀" if mode == "unit_cell" else "전체 구조",
            "유속":         _spd_txt,
            "영각":         _ang_txt,
            "총 케이스":    f"{len(speeds)*len(angles)}개" + (" (단일)" if len(speeds)*len(angles)==1 else ""),
            "MPI 코어":     f"{n_cores}개",
            "최대 반복":    str(end_time),
            "수렴 기준":    f"{residual_control:.0e}",
            "저장 간격":    f"{int(write_interval)} 스텝",
            "해수 밀도":    f"{rho:.1f} kg/m³",
            "동점성계수":   f"{nu:.2f} × 10⁻⁶ m²/s",
            "난류 강도":    f"{ti}%",
        }
        # 기준면적 Aref — 자동/직접 입력 어느 쪽이 쓰이는지 항상 명시
        _ovr = _effective_aref()
        if _ovr > 0:
            summary_data["기준면적 Aref"] = (
                f"{_ovr:.4e} m²  ({_ovr*1e4:.4f} cm²) — 사용자 직접 입력")
        else:
            _si = _auto_aref_info(mode)
            summary_data["기준면적 Aref"] = (
                f"{_si['value']:.4e} m²  ({_si['value']*1e4:.4f} cm²) — 자동: {_si['basis']}"
                if _si["value"] is not None else f"자동 — {_si['basis']}")
        if mode == "unit_cell":
            summary_data["단위 셀 크기"] = f"{cell_size*1000:.1f} mm"
            summary_data["고형률 Sn"] = f"{float(solidity):.3f}  (참고용 2d/a 추정)"
            summary_data["반복 수 (Nx×Ny)"] = (
                f"{ss.unit_nx} × {ss.unit_ny} (보고용 — 격자·Cd·시간 불변)")
            # 투영면적: 그물면 법선 투영(실측) × 반복수(Nx×Ny) = 총 투영면적 표시.
            # (해석용 Aref는 1셀 고정 — Cd는 nx/ny 불변. 표시값은 총면적.)
            if ss.get("stl_net_path"):
                try:
                    from cfd_manager import compute_projected_area as _cpa2
                    _aref2 = _cpa2(Path(ss.stl_net_path), (0.0, 0.0, 1.0))
                    _nxny2 = int(ss.unit_nx) * int(ss.unit_ny)
                    summary_data["투영면적"] = (
                        f"{_aref2*_nxny2:.4e} m²  (실측 그물면 법선 투영 × 반복수 {_nxny2}; "
                        f"1셀 {_aref2:.4e} m²)"
                    )
                except Exception:
                    pass
            summary_data["정밀화 레벨"] = f"{int(refine_level)} (min {max(1,int(refine_level)-1)} / max {int(refine_level)})"
            # 그물실 격자 해상도 — 망목이 커지면 실 대비 격자가 조용히 거칠어지므로
            # 항상 숫자로 보여준다(모르는 사이에 무너지지 않도록).
            _wd_mm = float(ss.get("auto_wire_d_mm") or 0.0)
            if _wd_mm > 0:
                _tr = twine_resolution(unit_cell_base_mm(cell_size), _wd_mm,
                                       int(refine_level))
                summary_data["그물실 해상도"] = (
                    f"{_tr['cells_per_d']:.1f} 셀/지름 "
                    f"(실 {_wd_mm:.2f} mm ÷ 최소셀 {_tr['finest_mm']:.3f} mm)"
                    + ("  ✅" if _tr["ok"]
                       else f"  ⚠️ 목표 {_tr['target']:.0f} 셀 미만"))
        else:
            summary_data["가두리 직경"] = f"{cage_d:.1f} m"
            summary_data["가두리 수심"] = f"{cage_h:.1f} m"

        for key, val in summary_data.items():
            col_k, col_v = st.columns([1, 1])
            col_k.markdown(f"**{key}**")
            col_v.markdown(val)

        st.divider()

        # ─── 예상 소요 시간 (조건에 따라 동적 갱신) ────────────────────────
        # 항목1·5: 입력 탭 추정과 실행 버튼 추정을 '동일한 중앙 로직'으로 통일한다.
        #   - 1케이스 추정 estimate_case_minutes = 전처리(메싱) + 솔버 시간(둘 다 포함)
        #   - 총 추정 estimate_total_minutes = 케이스당 추정 × 케이스 수(실측 보정 반영)
        # 입력 탭은 '총 예상 시간(전처리+솔버, 전 케이스)'을 표시해 실행 버튼과 일치시킨다.
        _rl_e = refine_level if mode == "unit_cell" else 3
        _n_cases = max(1, len(speeds) * len(angles))
        _est_total = estimate_total_minutes(mode, _n_cases, end_time, _rl_e, n_cores)
        # 항목2: breakdown(케이스당)을 '총/케이스수'로 산출 → 총합과 항상 일치
        # (모순 방지). 총 예상은 전처리(메싱)+솔버 시간 × 전 케이스를 반영한다.
        _est_per = _est_total / _n_cases
        st.markdown("### ⏱️ 예상 소요 시간")
        _ec1, _ec2 = st.columns([1, 1.4])
        # 요구서 §14: 정확히 예측할 수 없으면 단일값으로 표시하지 않는다.
        # 실측 보정을 거쳐도 수렴 반복 수는 사전 확정이 불가능하므로 범위로 낸다.
        _ec1.metric("총 예상 시간", f"{fmt_duration(_est_total*0.6)}~{fmt_duration(_est_total*1.8)}")
        _ec2.caption(
            f"중앙 추정 **{fmt_duration(_est_total)}** "
            f"(케이스 {_n_cases}개 × 약 {fmt_duration(_est_per)}/케이스, 전처리+솔버 포함). "
            f"반복 {int(end_time)} · 정밀화 {int(_rl_e)} · {n_cores}코어. "
            f"수렴 반복 수는 사전에 확정할 수 없어 **범위로 표시**합니다 — "
            f"수렴이 먼저 오면 하한보다 짧고, 미수렴이면 상한을 넘을 수 있습니다."
        )

        # ─── 예상 격자 규모·메모리 (요구서 §14) ────────────────────────────
        _stl_est = ss.get("stl_net_path") or ss.get("stl_cage_path")
        if _stl_est and Path(_stl_est).exists():
            try:
                _cdm = critical_dimension(Path(_stl_est))
                _spans_m = [s / 1000.0 for s in (_cdm.get("spans") or [0, 0, 0])]
                _area_m2 = compute_surface_area(Path(_stl_est))
                _rl_est = int(ss.get("refine_level_preset", 3))
                if mode == "unit_cell":
                    _a = float(ss.get("cell_size_mm", 20.0)) / 1000.0
                    _base_m = _uc_base_mm(_a) / 1000.0
                    _bg = (_a / _base_m) ** 2 * (2 * _a / _base_m)
                else:
                    _L = max(_spans_m) or 1.0
                    if ss.get("net_grid_redesign"):
                        _base_m = net_grid_base_cell(
                            (_cdm.get("bbox_min") or 1.0), _rl_est,
                            float(ss.get("net_grid_target_cells", 75.0)),
                            domain_m=(7.0 * _L, 4.0 * _L, 4.0 * _L))
                        _dom = (7.0 * _L, 4.0 * _L, 4.0 * _L)
                    else:
                        _base_m = _L / 8.0
                        _dom = (10.0 * _L, 6.0 * _L, 6.0 * _L)
                    _bg = (_dom[0] / _base_m) * (_dom[1] / _base_m) * (_dom[2] / _base_m)
                _est = estimate_mesh_size(_bg, _area_m2, _base_m / (2 ** _rl_est))
                if _est.get("ok"):
                    ss["_est_cells_max"] = _est["cells_max"]
                    ss["_est_mem_max_gb"] = _est["mem_max_gb"]
                    st.markdown("### 🧊 예상 격자 규모")
                    _bgs = (f"{_est['background_cells']/1e6:.2f}백만"
                            if _est['background_cells'] >= 1e5
                            else f"{_est['background_cells']:,.0f}개")
                    st.write(f"- 예상 셀 수: 약 **{_est['cells_min']/1e6:.2f} ~ "
                             f"{_est['cells_max']/1e6:.2f} 백만** (배경 "
                             f"{_bgs} + 표면 정밀화)")
                    st.write(f"- 예상 메모리: 약 **{_est['mem_min_gb']:.1f} ~ "
                             f"{_est['mem_max_gb']:.1f} GB** (격자 생성 시 최대)")
                    why("표면 정밀화 셀은 표면 주위 껍질에 생기므로 (표면적 ÷ 최소셀²)에 "
                        "비례합니다. 비례계수는 본 프로그램 실측 2건으로 보정했습니다 — "
                        "목표 75(배경 41.9만 → 최종 195.9만), 목표 150(배경 169.6만 → "
                        "최종 646.6만). 메모리는 1,189만 셀에서 19 GB 를 쓴 실측에서 "
                        "환산했습니다. 형상·정밀화 설정에 따라 달라지므로 범위로 냅니다.",
                        "왜? (셀 수 추정 근거)")
            except Exception as _ee:
                st.caption(f"예상 격자 규모를 계산하지 못했습니다({_ee}).")
        # 입력값(반복·정밀화·코어·nx/ny) 변경 시점 상태 저장
        _persist_input_state()

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

        # ─── 좌표계 / 영각 해석 (통일 표시 좌표계) ──────────────────
        from cfd_manager import display_convention, verify_coordinate_consistency
        st.markdown("### 🧭 좌표계 · 영각 해석 (통일 규약)")
        _a0 = float(angles[0]) if angles else 0.0
        _disp = display_convention(_a0)
        _fv = _disp["flow"]
        st.info(
            "**통일 표시 좌표계 — 두 모드 공통**\n\n"
            "- 전역 축: **+X = 그물면 법선(정면)**, **X–Y = 수면**, **+Z = 수심(↓)**\n"
            "- 그물 형상은 **Y–Z 평면**에 놓임(법선 +X)\n"
            "- 영각(AoA): **AoA=90° → 유속 −X→+X (정면)**, "
            "**AoA=0° → 유속 +Y→−Y (그레이징)**\n"
            f"- 유속 d(AoA) = (sinα, −cosα, 0)  →  현재 α={_a0:.1f}° 일 때 "
            f"**({_fv[0]:.3f}, {_fv[1]:.3f}, {_fv[2]:.3f})**\n"
            "- dragDir = 유속 방향, liftDir = 유속 수직(수면 내), pitchAxis = +Z(수심)\n\n"
            "*솔버 내부 좌표는 모드별로 수치 안정성에 맞춰 두되, 화면·리포트·시각화는 "
            "위 통일 좌표계로 변환해 표시합니다(저위험 프레임 통일).*")
        with st.expander("🔍 좌표 일관성 자동 검증 (솔버→표시 프레임 변환)", expanded=False):
            _rep = verify_coordinate_consistency(tol_deg=1.0)
            import pandas as _pd_cc
            _rows = []
            for _a, _r in _rep.items():
                _uf = _r["uc"]["flow_display"]; _ff = _r["fs"]["flow_display"]
                _rows.append({
                    "α [°]": _a,
                    "표시유속(목표)": f"({_r['display_flow'][0]:.2f},{_r['display_flow'][1]:.2f},0)",
                    "UC 변환유속": f"({_uf[0]:.2f},{_uf[1]:.2f},{_uf[2]:.2f})",
                    "FS 변환유속": f"({_ff[0]:.2f},{_ff[1]:.2f},{_ff[2]:.2f})",
                    "상대각차[°]": round(_r["rel_diff_deg"], 3),
                    "일치": "✅" if _r["ok"] else "⚠️",
                })
            st.dataframe(_pd_cc.DataFrame(_rows), use_container_width=True, hide_index=True)
            _warns = [w for _r in _rep.values() for w in _r["warnings"]]
            if _warns:
                for w in _warns:
                    st.warning(f"⚠️ {w}")
            else:
                st.success("좌표 통일 OK — 두 모드의 솔버 유속·법선이 모든 영각에서 통일 "
                           "표시 프레임 d(AoA)=(sinα,−cosα,0)·법선(+X)으로 정확히 변환됩니다.")

        with st.expander("📐 단위셀 ↔ 전체구조 Cd 차이는 왜 생기나 (물리적 원인)",
                         expanded=False):
            st.markdown(
                "두 모드는 **좌표계·기준면적(그물면)·정규화·난류모델·수치스킴이 모두 "
                "동일**하게 맞춰져 있습니다. 그럼에도 같은 조건의 Cd 절대값이 정확히 "
                "일치하지 않는 것은 **버그가 아니라 서로 다른 물리계를 풀기 때문**입니다.\n\n"
                "| 구분 | 단위 셀 (주기 BC) | 전체 구조 (개방) |\n"
                "|---|---|---|\n"
                "| 모델링 대상 | **무한 그물망** (셀이 사방 반복) | **유한 그물망** (3×3 등 실제 크기) |\n"
                "| 유동 | 전량이 그물을 통과 (우회 불가) | 가장자리로 우회 가능 |\n"
                "| 그물에 걸리는 힘 | **순수 법선 방향** (대칭) | 법선 + **유동방향 압력항**(전·후면 정체/후류) |\n"
                "| 기준면적 Aref | 1셀 그물면 | 전체 그물면(=셀수×1셀) |\n\n"
                "**정량 검증 (v10, U=1.0·90° 정면, onemesh3 ↔ net_mesh_3by3, "
                "단위셀 압력 BC 수정 후):**\n\n"
                "| 항목 | 단위셀 (무한 스크린) | 전체구조 (유한 3×3) |\n"
                "|---|---|---|\n"
                "| Cd (그물선 면적 기준) | **0.681** (도메인·격자 감도 ±0.2%) | **1.055** (격자 감도 −0.6%) |\n"
                "| 수렴 | SIMPLE 정식 수렴, 질량유량 오차 0 | SIMPLE 정식 수렴 |\n"
                "| 후류 결손 | **0%** (질량보존 — 우회 불가) | **12%** (전역 후류) |\n"
                "| 압력 점프 Δp/q | **0.567** (스크린 손실계수 K) | ~0 (개방역 압력 회복) |\n"
                "| 하중 분포 | (1셀) | 그물 전체 균일 (가장자리/중앙 = 0.93) |\n\n"
                "두 값의 차이(~35%)는 수치 오류가 아니라 **다른 물리계**의 결과입니다: "
                "무한 스크린(단위셀)은 유량이 강제 통과되며 압력강하가 개구부 **제트의 "
                "운동량**으로 하류에 수송되어 그물선 하중이 낮아지고, 유한 그물(전체구조)의 "
                "그물선은 우회·전역 압력장 속에서 **준자유류 원기둥**(Cd≈1.0–1.2)처럼 "
                "거동합니다. 배열이 커질수록(N×N→∞) 전체구조 내부는 단위셀 값에 수렴합니다.\n\n"
                "⚠️ **주의(v10):** 2026-07-02 이전의 단위셀 Cd 값은 압력 경계조건 "
                "결함(입·출구 뒤바뀜 → 질량유량 드리프트)이 있어 신뢰할 수 없습니다. "
                "템플릿 수정 후 새로 해석한 값만 사용하세요.\n\n"
                "**활용 가이드:** 대형 배열 내부·무한 균일 그물의 대표 계수가 필요하면 "
                "*단위 셀*을, 소형 조립체(3×3 등)의 실제 항력·가장자리 효과가 필요하면 "
                "*전체 구조*를 사용하세요. 둘은 상호 보완 관계이며 절대값 일치를 목표로 "
                "하지 않습니다.")

        st.divider()

        # ─── 해석 매트릭스 미리보기 ───────────────────────────────────────
        st.markdown("### 📊 해석 매트릭스")
        import pandas as _pd_mx
        _mx = {f"U={s:.2f}": ["○" for _ in angles] for s in speeds}
        _mx_df = _pd_mx.DataFrame(_mx, index=[f"α={a:.1f}°" for a in angles])
        st.dataframe(_mx_df, use_container_width=True,
                     height=min(38 + len(angles) * 35, 320))

        # ─── 결과 CSV 파일명 ──────────────────────────────────────────────
        _csv_default = f"force_coeffs_{mode}_{datetime.now():%Y%m%d}.csv"
        csv_name = st.text_input(
            "결과 CSV 파일명",
            value=_csv_default,
            key="batch_csv_name",
        )
        # 항목7: None/빈 값 방어 — 새 프로젝트 초기화 등으로 batch_csv_name 이 None 이면
        # PosixPath / None TypeError 가 발생하므로 기본값으로 폴백한다.
        if not csv_name or not str(csv_name).strip():
            csv_name = _csv_default
        # 활성 프로젝트가 있으면 결과 CSV 를 프로젝트 폴더 안에 둔다(항목1).
        csv_path = _project_csv_path(mode, csv_name)

        st.divider()

        # ─── 실행 버튼 (단일·배치 통합 — 단계수 1×1이면 단일) ──────────────
        st.markdown("### 🚀 해석 실행")

        run_disabled = (ss.job_status == "running")
        # 항목1·5: '예상 소요 시간' 섹션에서 계산한 값(_n_cases·_est_total)을 그대로
        # 재사용해 입력 탭 표시와 실행 버튼 표시가 항상 동일하도록 보장(중앙화).
        _ncase_run = _n_cases
        _run_label = ("▶️ 해석 시작 (단일)" if _ncase_run == 1
                      else f"🚀 배치 해석 시작 ({_ncase_run}개 · 예상 {fmt_duration(_est_total)})")

        # ── 실행 전 사전 점검 (요구서 §24·§25) ────────────────────────────
        # 치명적 항목이 하나라도 있으면 실행 버튼을 막는다. 각 항목은
        # 무엇이/왜/어떻게 세 가지를 함께 보여준다.
        _pf: List[Dict[str, str]] = []
        try:
            _stl_pf = (Path(ss.stl_net_path) if ss.get("stl_net_path")
                       else (Path(ss.stl_cage_path) if ss.get("stl_cage_path") else None))
            _pf = preflight_checks(
                mode, _stl_pf, speed=max(speeds) if speeds else 0.0,
                rho=float(ss.get("rho", 1025.0)), nu=_nu_si(),
                aref=(_effective_aref() if ss.get("aref_mode") == "직접 입력" else None),
                est_cells=ss.get("_est_cells_max"),
                est_mem_gb=ss.get("_est_mem_max_gb"),
                mem_limit_gb=_total_mem_gb(),
                turbulence=(ss.get("tr_turbulence") if str(ss.get("solver_mode", "")).startswith("Transient") else None),
                solver=("Transient" if str(ss.get("solver_mode", "")).startswith("Transient") else "Steady"))
        except Exception as _pe:
            st.caption(f"사전 점검을 수행하지 못했습니다({_pe}).")
        _pf_crit = [c for c in _pf if c["level"] == "critical"]
        _pf_warn = [c for c in _pf if c["level"] == "warning"]
        if _pf_crit or _pf_warn:
            with st.expander(
                    ("🚫 실행 전 점검 — 치명적 문제 "
                     f"{len(_pf_crit)}건, 주의 {len(_pf_warn)}건" if _pf_crit
                     else f"⚠️ 실행 전 점검 — 주의 {len(_pf_warn)}건"),
                    expanded=bool(_pf_crit)):
                for _c in _pf_crit + _pf_warn:
                    _fn = st.error if _c["level"] == "critical" else st.warning
                    _fn(f"**{_c['what']}**\n\n· 이유: {_c['why']}\n\n· 조치: {_c['how']}")

        # ── 검증 도구 (요구서 §19·§20) ────────────────────────────────────
        with st.expander("🔬 검증 도구 — 격자 독립성 · 문헌값 비교", expanded=False):
            st.markdown("**격자 독립성 시험 (§19)**")
            st.caption("같은 물리조건에서 정밀화 레벨만 바꿔 연속 실행하고 "
                       "셀 수·CD·CL 과 변화율을 비교합니다. 레벨 하나당 통상 "
                       "해석 1회와 같은 시간이 걸립니다.")
            _mi_lv = st.multiselect(
                "비교할 정밀화 레벨", [2, 3, 4, 5, 6, 7, 8],
                default=list(ss.get("mi_levels", [3, 4, 5])), key="_w_mi_levels")
            ss["mi_levels"] = _mi_lv
            _mi_disabled = (ss.job_status == "running") or len(_mi_lv) < 2 or bool(_pf_crit)
            if st.button("격자 독립성 시험 실행", disabled=_mi_disabled,
                         key="_btn_mi", use_container_width=True):
                _start_mesh_independence(mode, sorted(_mi_lv), n_cores, rho, ti)
            if ss.get("mi_rows"):
                _rows = [{"레벨": r["level"],
                          "셀 수": f"{r['cells']:,}" if r.get("cells") else "—",
                          "CD": "—" if r.get("Cd") is None else f"{r['Cd']:.5f}",
                          "CL": "—" if r.get("Cl") is None else f"{r['Cl']:.5f}",
                          "CD 변화율": "—" if r.get("dCd_pct") is None else f"{r['dCd_pct']:+.2f}%",
                          "CL 변화율": "—" if r.get("dCl_pct") is None else f"{r['dCl_pct']:+.2f}%"}
                         for r in ss["mi_rows"]]
                st.dataframe(_rows, use_container_width=True, hide_index=True)
                _last = [r for r in ss["mi_rows"] if r.get("dCd_pct") is not None]
                if _last:
                    _d = abs(_last[-1]["dCd_pct"])
                    if _d < 2.0:
                        st.success(f"가장 촘촘한 두 격자의 CD 변화가 {_d:.2f}% 로 "
                                   "격자 독립성이 확보된 것으로 볼 수 있습니다.")
                    else:
                        st.warning(f"가장 촘촘한 두 격자의 CD 변화가 {_d:.2f}% 입니다 — "
                                   "아직 격자에 의존합니다. 레벨을 더 올려 보십시오.")
                why("격자 독립성은 '격자를 더 촘촘히 해도 값이 바뀌지 않는 상태'를 "
                    "확인하는 절차입니다. 변화율은 셀 수 오름차순으로 직전 격자 대비 "
                    "계산합니다. 통상 2% 이내면 독립성이 확보된 것으로 보지만, 절대 "
                    "기준은 아니며 목적하는 정확도에 따라 달라집니다.",
                    "왜? (격자 독립성 판정 기준)")

            st.divider()
            st.markdown("**검증용 Reference case (§20)**")
            st.caption("문헌값이 있는 기본 형상을 같은 파이프라인으로 돌려 프로그램 "
                       "자체를 점검합니다. 문헌값은 출처와 함께 저장돼 있습니다.")
            _rc_key = st.selectbox("검증 형상", list(REFERENCE_CASES.keys()),
                                   format_func=lambda k: REFERENCE_CASES[k]["label"],
                                   key="ref_case_key")
            _rc = REFERENCE_CASES[_rc_key]
            _lo, _hi = _rc["Re_range"]
            st.write(f"- 문헌값 **CD = {_rc['Cd_ref']}** · 적용 범위 Re "
                     f"{_lo:.0e} ~ {_hi:.0e} · 대표 길이 {_rc['length_m']*1000:.0f} mm")
            st.caption(f"출처: {_rc['source']}")
            if _rc.get("note"):
                st.info(_rc["note"])
            _rc_u = st.number_input("검증 유속 [m/s]", value=float(ss.get("ref_speed", 1.0)),
                                    min_value=0.01, max_value=10.0, step=0.1,
                                    key="ref_speed")
            _rc_re = _rc_u * _rc["length_m"] / _nu_si()
            st.write(f"- 이 조건의 Reynolds 수 = **{_rc_re:.2e}** "
                     + ("✅ 문헌값 적용 범위 안" if _lo <= _rc_re <= _hi
                        else "⚠️ 문헌값 적용 범위 밖 — 비교가 성립하지 않을 수 있습니다"))
            if st.button("검증 케이스 실행", disabled=(ss.job_status == "running"),
                         key="_btn_ref", use_container_width=True):
                _start_reference_case(_rc_key, _rc_u, n_cores, rho, ti)
            if ss.get("ref_result"):
                _r = ss["ref_result"]
                st.dataframe([{"항목": "CFD 결과", "값": f"{_r['Cd_cfd']:.4f}"},
                              {"항목": "문헌값", "값": f"{_r['Cd_ref']:.4f}"},
                              {"항목": "상대오차", "값": f"{_r['error_pct']:+.1f}%"},
                              {"항목": "Reynolds", "값": f"{_r['Re']:.2e}"}],
                             use_container_width=True, hide_index=True)
                st.caption(f"출처: {_r['source']}")

        btn_col1, btn_col2 = st.columns(2)

        # 항목9: 실행 전 이미 완료된 조건과의 중복 검사(덮어쓰기 경고용)
        _run_conds = {(round(float(s), 2), round(float(a), 1))
                      for s in speeds for a in angles}
        _overlap_conds = sorted(_run_conds & _scan_done_conditions(mode))

        with btn_col1:
            if st.button(_run_label, disabled=(run_disabled or bool(_pf_crit)),
                         use_container_width=True, type="primary",
                         help=("사전 점검에서 치명적 문제가 발견돼 실행이 막혀 "
                               "있습니다. 위 점검 항목을 해결하십시오."
                               if _pf_crit else None)):
                if _overlap_conds:
                    # 완료 조건 중복 → 즉시 실행하지 않고 덮어쓰기 경고 표시
                    ss["_ovw_overlap"] = _overlap_conds
                    st.rerun()
                else:
                    ss["_ovw_overlap"] = None
                    _start_batch_analysis(
                        mode, speeds, angles, csv_path, n_cores, rho, ti,
                        nx=ss.unit_nx, ny=ss.unit_ny,
                    )

        with btn_col2:
            if st.button(
                "⏹️ 해석 중지",
                disabled=(ss.job_status != "running"),
                use_container_width=True,
                type="secondary"
            ):
                _stop_analysis()

        # 항목9: 덮어쓰기 경고 + 확인. 기존 완료 결과가 있는 조건을 재해석할 때
        # 사용자가 명시적으로 동의해야 진행(진행 시 케이스 디렉토리·CSV 행 모두 교체).
        if ss.get("_ovw_overlap"):
            _ct = ", ".join(f"U={s:.2f}·α={a:.1f}°" for s, a in ss["_ovw_overlap"])
            st.warning(
                f"⚠️ 조건 '{_ct}'에 이미 해석 결과가 있습니다. "
                "계속하면 기존 결과를 덮어씁니다.")
            _w1, _w2 = st.columns(2)
            with _w1:
                if st.button("✅ 덮어쓰기하고 진행", type="primary",
                             use_container_width=True, key="ovw_go",
                             disabled=run_disabled):
                    ss["_ovw_overlap"] = None
                    _start_batch_analysis(
                        mode, speeds, angles, csv_path, n_cores, rho, ti,
                        nx=ss.unit_nx, ny=ss.unit_ny,
                    )
            with _w2:
                if st.button("취소", use_container_width=True, key="ovw_cancel"):
                    ss["_ovw_overlap"] = None
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  탭 4: 결과 분석
# ═══════════════════════════════════════════════════════════════════════════

with tab_results:
    st.markdown("### 📈 해석 결과 분석")

    import re as _re_res

    # ─── 케이스 스캔 (해석중_/해석완료_ 접두어로 상태 판정) ───────────────────
    # 활성 프로젝트가 있으면 그 폴더만 스캔(프로젝트 격리, 항목1·5).
    def _scan_result_cases(_md):
        _root = _results_root(_md)
        _out = []
        if not _root.exists():
            return _out
        for _d in sorted(_root.iterdir(),
                         key=lambda x: x.stat().st_mtime, reverse=True):
            if not _d.is_dir() or not (_d / "constant").exists():
                continue
            _m = _re_res.search(r"_U([0-9.]+)_A(-?[0-9.]+)", _d.name)
            if not _m:
                continue
            _s, _a = float(_m.group(1)), float(_m.group(2))
            _tdirs = [t for t in _d.iterdir()
                      if t.is_dir() and t.name not in ("0",)
                      and t.name.replace(".", "", 1).isdigit()]
            _has_post = (_d / "postProcessing").exists()
            _run_pref = _d.name.startswith("해석중_")
            _done_pref = _d.name.startswith("해석완료_")
            if _run_pref and ss.job_status == "running":
                _stt = "running"
            elif _done_pref:
                _stt = "done"
            elif _run_pref:
                _stt = "stopped"
            elif _tdirs and _has_post:
                _stt = "partial"
            elif _tdirs or _has_post:
                _stt = "partial"
            else:
                _stt = "mesh"
            _out.append(dict(path=_d, name=_d.name, speed=_s, angle=_a,
                             status=_stt, has_fields=bool(_tdirs)))
        return _out

    _cases = _scan_result_cases(mode)

    # ── 조건별 상태(3상태): 완료(☑)/진행중(⏳)/미해석(☐) — 항목 9 ─────────
    # 완료/부분(필드 보유)=완료, 해석중_(실행중)=진행중, 그 외는 매트릭스상 미해석.
    _by_cond = {}
    _mprio = {"running": 3, "done": 2}
    for _c in _cases:
        # 완료 판정: '해석완료_' 접두어(status==done)는 솔버 종료 + Cd 추출 성공
        # 시점에만 부여되므로 신뢰할 수 있는 '해석 완료' 신호다. 과거엔 has_fields
        # (reconstruct된 시간 디렉토리)까지 요구했으나, full_structure 등에서
        # reconstructPar 가 실패해 상위 시간 디렉토리가 없으면(병렬 데이터만 존재)
        # 완료된 케이스인데도 '해석 전'으로 숨겨지는 문제가 있었다(항목3).
        # → 완료(☑)는 status==done 만으로 등록하고, 3D 재구성 데이터 유무는
        #   유동장 표시 단계에서 별도로 처리한다(필드 없으면 안내 + CSV 결과 제공).
        if _c["status"] == "done":
            _mst = "done"
        elif _c["status"] == "running":
            _mst = "running"
        else:
            continue
        _k = (round(_c["speed"], 2), round(_c["angle"], 1))
        if _k not in _by_cond or _mprio[_mst] > _mprio[_by_cond[_k]["mstatus"]]:
            _by_cond[_k] = dict(case=_c, mstatus=_mst)

    # ── 매트릭스 그리드 = 입력 설정의 유속·영각 단계만 (세션 키 연동) ──────────
    _um = float(ss.get("u_min", 1.0)); _ux = float(ss.get("u_max", 1.0))
    _us = int(ss.get("u_steps", 1))
    _amn = float(ss.get("a_min", 0.0)); _amx = float(ss.get("a_max", 0.0))
    _asp = int(ss.get("a_steps", 1))
    # 중복 제거: min==max 인데 단계수>1 이면 linspace 가 같은 값을 반복 생성해
    # 매트릭스 버튼 키(mx_<s>_<a>)가 중복 → StreamlitDuplicateElementKey 발생.
    # 정렬된 고유값만 사용해 조건당 셀 1개를 보장한다.
    _speeds_grid = sorted({round(float(s), 2)
                           for s in np.linspace(_um, _ux, max(1, _us))})
    _angles_grid = sorted({round(float(a), 1)
                           for a in np.linspace(_amn, _amx, max(1, _asp))})
    # 항목2: 체크표시 대신 명시적 상태 텍스트 라벨.
    _BOX = {"done": "해석 완료", "running": "해석 중", "none": "해석 전"}
    _MTXT = {"done": "완료", "running": "진행 중", "none": "미해석"}

    if not _speeds_grid or not _angles_grid:
        st.info("입력 설정에서 유속·영각 조건을 먼저 지정하세요.")
    else:
        # ─── 해석 매트릭스 ────────────────────────────────────────────────
        st.markdown("#### 🧮 해석 매트릭스")
        st.caption("'해석 완료' / '해석 중' / '해석 전'   ·   셀 클릭 → 유동장 표시"
                   "(선택 셀은 색 반전으로 강조). "
                   "행·열은 **입력 설정의 유속·영각 단계**와 일치합니다.")

        _hdr = st.columns([0.9] + [1] * len(_speeds_grid))
        _hdr[0].markdown("**α \\ U**")
        for _j, _s in enumerate(_speeds_grid):
            _hdr[_j + 1].markdown(f"**U={_s:.2f}**")
        for _a in _angles_grid:
            _row = st.columns([0.9] + [1] * len(_speeds_grid))
            _row[0].markdown(f"**α={_a:.1f}°**")
            for _j, _s in enumerate(_speeds_grid):
                _ent = _by_cond.get((_s, _a))
                _mst = _ent["mstatus"] if _ent else "none"
                _cond_key = f"{_s:.2f}_{_a:.1f}"
                # 항목7: 현재 선택된 조건 셀은 색 반전(primary)으로 활성 표시.
                _is_sel = (ss.get("res_sel_cond") == _cond_key)
                with _row[_j + 1]:
                    if st.button(_BOX[_mst], key=f"mx_{_cond_key}",
                                 help=f"U={_s:.2f}, α={_a:.1f}° — {_MTXT[_mst]}",
                                 use_container_width=True,
                                 type=("primary" if _is_sel else "secondary")):
                        ss.res_sel_cond = _cond_key
                        st.rerun()

        # ── 입력 설정 밖의 디스크 완료 케이스 별도 표시 ──────────────────────
        _grid_keys = {(s, a) for a in _angles_grid for s in _speeds_grid}
        _disk_extra = {k: v for k, v in _by_cond.items() if k not in _grid_keys}
        if _disk_extra:
            with st.expander(f"📂 이전 완료 케이스 ({len(_disk_extra)}개) — 클릭해서 유동장 보기",
                             expanded=False):
                _de_speeds = sorted({k[0] for k in _disk_extra})
                _de_angles = sorted({k[1] for k in _disk_extra})
                _de_hdr = st.columns([0.9] + [1] * len(_de_speeds))
                _de_hdr[0].markdown("**α \\ U**")
                for _dj, _ds in enumerate(_de_speeds):
                    _de_hdr[_dj + 1].markdown(f"**U={_ds:.2f}**")
                for _da in _de_angles:
                    _de_row = st.columns([0.9] + [1] * len(_de_speeds))
                    _de_row[0].markdown(f"**α={_da:.1f}°**")
                    for _dj, _ds in enumerate(_de_speeds):
                        _de_ent = _disk_extra.get((_ds, _da))
                        _de_mst = _de_ent["mstatus"] if _de_ent else "none"
                        _de_key = f"{_ds:.2f}_{_da:.1f}"
                        _de_sel = (ss.get("res_sel_cond") == _de_key)
                        with _de_row[_dj + 1]:
                            if st.button(_BOX[_de_mst],
                                         key=f"de_{_de_key}",
                                         help=f"U={_ds:.2f}, α={_da:.1f}° — {_MTXT[_de_mst]}",
                                         use_container_width=True,
                                         type=("primary" if _de_sel else "secondary")):
                                ss.res_sel_cond = _de_key
                                st.rerun()

        st.divider()

        # ── 선택 조건 해석 (기본: 첫 완료/진행 셀, 없으면 그리드 첫 셀) ──────
        # 저장된 선택이 현재 그리드 밖이면(입력 조건 변경 등) 재기본화한다.
        # _grid_keys는 매트릭스 + 디스크 별도 섹션 합집합으로 유효 범위 확장.
        _grid_keys = {(s, a) for a in _angles_grid for s in _speeds_grid} | set(_disk_extra.keys())
        _cur = ss.get("res_sel_cond")
        _cur_ok = False
        if _cur:
            try:
                _cs, _ca = _cur.split("_")
                _cur_ok = (round(float(_cs), 2), round(float(_ca), 1)) in _grid_keys
            except Exception:
                _cur_ok = False
        if not _cur_ok:
            _dflt = next((f"{s:.2f}_{a:.1f}" for a in _angles_grid
                          for s in _speeds_grid if (s, a) in _by_cond), None)
            ss.res_sel_cond = _dflt or f"{_speeds_grid[0]:.2f}_{_angles_grid[0]:.1f}"
        selected_case_dir = None
        _sel_case = None; _sel_mst = "none"; _sel_s = None; _sel_a = None
        try:
            _ps, _pa = ss.res_sel_cond.split("_")
            _sel_s, _sel_a = round(float(_ps), 2), round(float(_pa), 1)
            _ent = _by_cond.get((_sel_s, _sel_a))
            if _ent:
                _sel_case = _ent["case"]; _sel_mst = _ent["mstatus"]
                selected_case_dir = _sel_case["path"]
        except Exception:
            pass

        # ─── 시각화 탭 ────────────────────────────────────────────────────
        # 선택한 케이스의 신뢰성 등급 — 어느 하위 탭을 보든 항상 눈에 띄도록
        # 탭 위에 표시한다.
        if selected_case_dir is not None:
            _app_sel = ""
            try:
                _ct = (Path(selected_case_dir) / "system" / "controlDict").read_text()
                _ma = re.search(r"application\s+(\w+);", _ct)
                _app_sel = _ma.group(1) if _ma else ""
            except Exception:
                pass
            # ── 결과 신뢰성 판정 (요구서 §18) ─────────────────────────
            # 'Success' 한 마디로 끝내지 않고 무엇이 확인됐고 무엇이 미확인
            # 인지 등급과 사유로 남긴다.
            try:
                _ad_for_grade = None
                _stl_g = ss.get("stl_net_path") or ss.get("stl_cage_path")
                if _stl_g and Path(_stl_g).exists():
                    _cdg = critical_dimension(Path(_stl_g))
                    _critg = float(_cdg.get("bbox_min") or 0.0)
                    _spg = _cdg.get("spans") or [1.0]
                    _rlg = int(ss.get("refine_level_preset", 3))
                    _baseg = (unit_cell_base_mm(float(ss.get("cell_size_mm", 20.0))/1000.0)
                              if mode == "unit_cell" else max(_spg) / 8.0)
                    if mode == "full_structure" and ss.get("net_grid_redesign"):
                        _Lg = max(_spg) / 1000.0
                        _baseg = net_grid_base_cell(
                            _critg, _rlg, float(ss.get("net_grid_target_cells", 75.0)),
                            domain_m=(7.0*_Lg, 4.0*_Lg, 4.0*_Lg)) * 1000.0
                    _boxg = (_rlg - 1 if _rlg <= 3 else 2)
                    if ss.get("wake_box_mode") == "직접 지정":
                        _boxg = min(_rlg, int(ss.get("wake_box_level", 2)))
                    _famg = ("LES" if TRANSIENT_TURBULENCE_MODELS.get(
                                ss.get("tr_turbulence", "kOmegaSST")) == "LES"
                             and (_app_sel == 'pimpleFoam') else "RAS")
                    if _critg > 0 and _baseg > 0:
                        _ad_for_grade = mesh_adequacy(_critg, _baseg, _rlg, _boxg, _famg)
                _grade = result_reliability(
                    Path(selected_case_dir), adequacy=_ad_for_grade,
                    mesh_independence=bool(ss.get("mi_done")),
                    reference_checked=bool(ss.get("ref_done")))
                _badge = {"green": st.success, "yellow": st.warning,
                          "red": st.error}[_grade["grade"]]
                _badge(f"**Result status: {_grade['label']}**")
                if _grade["red"] or _grade["yellow"]:
                    with st.expander("판정 사유", expanded=(_grade["grade"] == "red")):
                        for _r in _grade["red"]:
                            st.markdown(f"- 🔴 {_r}")
                        for _r in _grade["yellow"]:
                            st.markdown(f"- 🟡 {_r}")
                st.dataframe(_grade["checks"], use_container_width=True,
                             hide_index=True)
                why("등급은 다음을 종합합니다 — 기준면적 기록 여부, 힘 계수 수렴"
                    "(후반 구간의 표류·진동), 격자 적정성(표면·후류 셀 수), "
                    "비정상 해석이면 진동 포착 여부, 격자 독립성·문헌 비교 수행 "
                    "여부. 하나라도 결과를 무의미하게 만드는 항목이 있으면 RED, "
                    "주의가 필요하면 YELLOW 입니다. 미실시 항목은 '틀렸다'가 "
                    "아니라 '확인되지 않았다'는 뜻입니다.", "왜? (신뢰성 등급 기준)")
            except Exception as _ge:
                st.caption(f"신뢰성 판정을 계산하지 못했습니다({_ge}).")

        r_tab1, r_tab2, r_tab3, r_tab5, r_tab4 = st.tabs([
            "🌊 유동장", "📉 수렴 이력", "📊 유속 감쇠",
            "⏱️ 시간이력 (비정상)", "💾 CSV 데이터"
        ])

        # ─── 비정상 해석 결과: 시간이력 · 통계 · steady 비교 (지시서 §15·§16) ──
        with r_tab5:
            if selected_case_dir is None:
                st.info("케이스를 먼저 선택하세요 (위 매트릭스에서 셀 클릭).")
            else:
                # 이 케이스가 실제로 pimpleFoam 으로 돌았는지 판별한다.
                # simpleFoam 의 이력은 '반복 횟수'라 물리시간이 아니며, 그 변동은
                # 수렴 드리프트일 뿐 물리적 비정상성이 아니다. 구분하지 않으면
                # 정상해석 결과를 비정상 결과로 오독하게 된다.
                _app = ""
                try:
                    _ctrl_txt = (Path(selected_case_dir) / "system" / "controlDict").read_text()
                    _mapp = re.search(r"application\s+(\w+);", _ctrl_txt)
                    _app = _mapp.group(1) if _mapp else ""
                except Exception:
                    pass
                _is_transient_case = (_app == "pimpleFoam")

                _hist = read_force_history(Path(selected_case_dir))
                if len(_hist["time"]) < 3:
                    st.info(
                        "이 케이스에는 시간이력이 없습니다.\n\n"
                        "**입력 설정 → Solver → Transient (pimpleFoam)** 으로 "
                        "해석해야 이 탭이 채워집니다.")
                elif not _is_transient_case:
                    st.warning(
                        f"⚠️ 이 케이스는 **정상상태 해석**입니다 (application = "
                        f"`{_app or '알 수 없음'}`).\n\n"
                        "아래 이력의 가로축은 물리시간이 아니라 **반복 횟수**이고, "
                        "그 변동은 물리적 비정상성이 아니라 **수렴 과정의 드리프트**입니다. "
                        "시간평균·RMS·비정상성 판정은 의미가 없으므로 표시하지 않습니다.")
                    try:
                        import plotly.graph_objects as _go
                        _f0 = _go.Figure()
                        for _nm, _col in (("Cd", "#c0392b"), ("Cl", "#2471a3")):
                            _f0.add_trace(_go.Scatter(x=_hist["time"], y=_hist[_nm],
                                                      mode="lines", name=_nm,
                                                      line=dict(color=_col, width=1.5)))
                        _f0.update_layout(height=320, xaxis_title="반복 횟수 [-]",
                                          yaxis_title="계수",
                                          margin=dict(l=50, r=30, t=20, b=40))
                        st.plotly_chart(_f0, use_container_width=True, key="r5_steady_hist")
                        st.caption("참고용 수렴 이력 — 마지막 값이 곧 정상해입니다.")
                    except Exception:
                        pass
                else:
                    _t_end = max(_hist["time"])
                    _t_min = min(_hist["time"])
                    _dflt = float(ss.get("tr_avg_start", 0.0)) or (_t_min + (_t_end - _t_min) * 0.5)
                    _avg_start = st.slider(
                        "평균 시작 시각 TavgStart", min_value=float(_t_min),
                        max_value=float(_t_end), value=float(min(max(_dflt, _t_min), _t_end)),
                        key="r5_avg_start",
                        help="이 시각 이후 구간만 시간평균합니다(초기 과도구간 제외).")
                    _stats = compute_transient_stats(Path(selected_case_dir),
                                                     t_avg_start=_avg_start)

                    # ── 시간이력 그래프 (평균구간 음영 + 평균선) ──
                    try:
                        import plotly.graph_objects as _go
                        from plotly.subplots import make_subplots as _msp
                        _fig = _msp(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=("Cd vs Time", "Cl vs Time"),
                                    vertical_spacing=0.12)
                        for _i, (_nm, _col) in enumerate([("Cd", "#c0392b"), ("Cl", "#2471a3")], 1):
                            _fig.add_trace(_go.Scatter(
                                x=_hist["time"], y=_hist[_nm], mode="lines",
                                name=_nm, line=dict(color=_col, width=1.5)), row=_i, col=1)
                            _m = _stats.get(f"mean_{_nm}")
                            if _m is not None:
                                _fig.add_hline(y=_m, line=dict(color=_col, dash="dash", width=1),
                                               row=_i, col=1,
                                               annotation_text=f"mean {_m:.4f}",
                                               annotation_position="right")
                            # 평균 구간 음영
                            _fig.add_vrect(x0=_avg_start, x1=_t_end,
                                           fillcolor="#f1c40f", opacity=0.12,
                                           line_width=0, row=_i, col=1)
                        _fig.update_xaxes(title_text="Time [s]", row=2, col=1)
                        _fig.update_layout(height=520, showlegend=False,
                                           margin=dict(l=50, r=30, t=50, b=40))
                        st.plotly_chart(_fig, use_container_width=True, key="r5_hist")
                        st.caption("🟨 음영 = 시간평균 구간 · ⌐ 파선 = 평균값")
                    except Exception as _e:
                        st.warning(f"그래프 생성 실패: {_e}")

                    # ── 통계 ──
                    st.markdown("#### 📊 평균구간 통계")
                    _c = st.columns(4)
                    for _i, _nm in enumerate(("Cd", "Cl")):
                        if f"mean_{_nm}" in _stats:
                            _c[_i * 2].metric(f"평균 {_nm}", f"{_stats[f'mean_{_nm}']:.5f}")
                            _c[_i * 2 + 1].metric(f"RMS {_nm}", f"{_stats[f'rms_{_nm}']:.5f}",
                                                  delta=f"±{_stats[f'std_{_nm}']:.5f} (표준편차)",
                                                  delta_color="off")
                    _rows_st = []
                    for _nm in ("Cd", "Cl", "Fx", "Fy", "Fz", "Ftotal"):
                        if f"mean_{_nm}" in _stats:
                            _u = " [N]" if _nm.startswith("F") else ""
                            _rows_st.append({
                                "항목": _nm + _u,
                                "평균": f"{_stats[f'mean_{_nm}']:.6g}",
                                "표준편차": f"{_stats[f'std_{_nm}']:.6g}",
                                "RMS": f"{_stats[f'rms_{_nm}']:.6g}",
                                "최소": f"{_stats[f'min_{_nm}']:.6g}",
                                "최대": f"{_stats[f'max_{_nm}']:.6g}",
                            })
                    if _rows_st:
                        st.dataframe(_rows_st, use_container_width=True, hide_index=True)
                    st.caption(f"평균구간 {_avg_start:.4g} ~ {_stats.get('t_end', _t_end):.4g} s · "
                               f"샘플 {_stats.get('n_samples', 0)}개")

                    # ── 보완②: 비정상성 포착 판정 ──
                    _uns = _stats.get("unsteadiness_Cd")
                    # 양력 진동 정보(그물처럼 항력 변동이 상쇄되는 형상의 판정 근거)
                    _fcl = _stats.get("freq_Cl")
                    _ucl = _stats.get("unsteadiness_Cl")
                    if _fcl:
                        st.caption(
                            f"양력 진동: 주파수 **{_fcl:.1f} Hz** "
                            f"(주기 {_stats.get('period_Cl', 0):.4g} s) · "
                            f"평균선 교차 {_stats.get('n_cross_Cl', 0)}회 · "
                            f"진폭/Cd평균 {(_ucl or 0)*100:.2f}%")
                    if _uns is not None:
                        if _stats.get("is_unsteady"):
                            _by = _stats.get("unsteady_by")
                            if _by == "Cl 진동":
                                st.success(
                                    f"✅ 비정상성 포착됨 — **양력 진동**으로 판정 "
                                    f"(주파수 {_fcl:.1f} Hz, 진폭/Cd평균 "
                                    f"{(_ucl or 0)*100:.2f}%). Cd 변동/평균은 "
                                    f"{_uns*100:.2f}% 로 작지만, 그물처럼 다수의 가는 "
                                    "요소로 된 형상은 각 요소의 방출 위상이 상쇄돼 "
                                    "합력 항력의 변동이 원래 작습니다. 이 경우 항력 "
                                    "변동이 아니라 양력 주파수로 판정하는 것이 맞습니다.")
                            else:
                                st.success(
                                    f"✅ 비정상성 포착됨 — Cd 변동/평균 = **{_uns*100:.2f}%** (≥1%). "
                                    "시간평균값을 정상해와 비교할 수 있습니다.")
                        else:
                            st.error(
                                f"⚠️ **비정상성 미포착** — Cd 변동/평균 = {_uns*100:.2f}% (<1%).\n\n"
                                "이 결과를 '정상해석으로 충분하다'는 근거로 쓰면 안 됩니다. "
                                "표준 kOmegaSST URANS 가 와류 방출을 감쇠시킨 것일 수 있습니다. "
                                "**난류모델을 kOmegaSSTDDES 로 바꿔 재검증**하세요(지시서 보완②).")

                    # ── 지시서 §16: steady ↔ transient 비교 ──
                    st.markdown("#### ⚖️ 정상 ↔ 비정상 비교")
                    _cd_steady = None
                    try:
                        import pandas as _pdc
                        for _csv in sorted(Path(_results_root(mode)).glob("*.csv")):
                            _dfc = _pdc.read_csv(_csv)
                            if {"speed_m_s", "angle_deg", "Cd"}.issubset(_dfc.columns) \
                                    and _sel_s is not None:
                                _hit = _dfc[(_dfc.speed_m_s.round(2) == round(_sel_s, 2))
                                            & (_dfc.angle_deg.round(1) == round(_sel_a, 1))]
                                if not _hit.empty:
                                    _cd_steady = float(_hit.iloc[-1]["Cd"])
                                    break
                    except Exception:
                        pass
                    _cd_tr = _stats.get("mean_Cd")
                    if _cd_steady is not None and _cd_tr is not None:
                        _diff = (_cd_tr - _cd_steady) / _cd_steady * 100
                        _k1, _k2, _k3 = st.columns(3)
                        _k1.metric("SimpleFoam CD", f"{_cd_steady:.5f}")
                        _k2.metric("PimpleFoam Mean CD", f"{_cd_tr:.5f}")
                        _k3.metric("Difference", f"{_diff:+.1f} %",
                                   help="(mean_CD_pimple − CD_simple) / CD_simple × 100")
                    else:
                        st.caption("같은 조건의 정상상태 Cd 를 CSV 에서 찾지 못해 "
                                   "비교를 생략했습니다.")

        with r_tab1:
            if _sel_s is not None:
                st.markdown(
                    f"**선택 조건:** U={_sel_s:.2f} m/s, α={_sel_a:.1f}°  —  "
                    f"{_BOX[_sel_mst]} {_MTXT[_sel_mst]}"
                    + (f"  ·  `{_sel_case['name']}`" if _sel_case else ""))

            _tnx = int(ss.get("unit_nx", 1)) if mode == "unit_cell" else 1
            _tny = int(ss.get("unit_ny", 1)) if mode == "unit_cell" else 1

            _has_3d = bool(_sel_case and _sel_case.get("has_fields"))
            _is_done_or_run = _sel_mst in ("done", "running") and selected_case_dir is not None
            if _is_done_or_run and not _has_3d:
                # 항목3: 해석은 완료(또는 진행)됐으나 3D 재구성 데이터가 없는 경우
                # — '해석 전'이 아니라 완료 상태를 명확히 안내하고 CSV 결과로 유도.
                st.markdown("""
                <div style="background:#e9f7ef;border:2px solid #0f9d58;
                            border-radius:10px;padding:32px;text-align:center;">
                    <h3 style="color:#0f9d58;">✅ 해석 완료 — 3D 재구성 데이터 없음</h3>
                    <p style="color:#444;">이 조건은 해석이 완료되어 <b>Cd/Cl 결과가
                    'CSV 데이터' 탭</b>에 있습니다. 다만 reconstructPar 단계가 수행되지
                    않아 유동장 3D(슬라이스·등치면)는 표시할 수 없습니다.</p>
                </div>""", unsafe_allow_html=True)
            elif not _is_done_or_run or selected_case_dir is None:
                # 미해석/결과 없음 — 빈 화면 + 안내
                st.markdown("""
                <div style="background:#f0f8ff;border:2px dashed #1a73e8;
                            border-radius:10px;padding:40px;text-align:center;">
                    <h3 style="color:#1a73e8;">🌊 해석 시작 전입니다</h3>
                    <p style="color:#666;">이 조건은 아직 유동장 결과가 없습니다.
                    '입력 설정'에서 해석을 시작하거나, 완료/진행 중(해석 완료/해석 중) 셀을 선택하세요.</p>
                </div>""", unsafe_allow_html=True)
            else:
                _vmode = st.radio(
                    "표시 방식", ["슬라이스", "볼륨(연속)", "입체", "등치면(스윕)"],
                    horizontal=True, key="r1_viewmode",
                    help="슬라이스: 단면  |  볼륨(연속): 형상 주변+후류를 연속 반투명 "
                         "볼륨으로 보간 표시  |  입체: 등치면 정적 3D(마우스로 회전·확대)  |  "
                         "등치면(스윕): 등치값을 자동으로 훑는 애니메이션")
                _r1_field = st.selectbox(
                    "시각화 필드", ["U", "p", "k", "omega"],
                    format_func=lambda x: {"U": "속도 |U|", "p": "압력 p",
                                           "k": "난류 k", "omega": "비소산율 ω"}[x],
                    key="r1_field")
                # 항목4: 모든 표시 모드에서 STL 형상을 결과와 함께 렌더링하고,
                # 그 가시성(불투명도)을 사용자가 조절하도록 STL 투명도 슬라이더 제공.
                # v13 항목13: 모든 투명도 슬라이더를 0~100%(0=완전 투명,
                # 100=완전 불투명)로 통일. 내부 렌더 API 는 0~1 을 쓰므로 /100.
                _stl_pct = st.slider(
                    "STL 형상 투명도 [%]", 0, 100,
                    int(round(float(ss.get("r1_stl_opacity", 0.15)) * 100)),
                    5, key="r1_stl_opacity_pct",
                    help="그물망(STL) 형상의 불투명도(0%=숨김, 100%=불투명). "
                         "모든 모드에 적용됩니다.")
                _stl_op = _stl_pct / 100.0
                ss["r1_stl_opacity"] = _stl_op
                _viz_r1 = CFDVisualizer(selected_case_dir)

                # 카메라 유지: 슬라이스·입체·등치면이 모두 동일 uirevision('flowfield')을
                # 쓰므로, 단 하나의 플래그로 "최초 1회만" camera를 figure에 넣는다.
                # 이후 모든 렌더(위젯 변경 + 표시 방식 전환 포함)에서 camera를 빼서
                # plotly.js가 사용자의 확대/회전 상태를 보존하게 한다. 표시 방식을
                # 바꿔도 플래그를 리셋하지 않으므로 확대 상태가 유지된다.
                _init_cam = not ss.get("_cam_init_field", False)
                # if "_cam_initialized" not in ss:
                #     ss["_cam_initialized"] = {}

                # _cam_key = (
                #     f"{selected_case_dir}_"
                #     f"{_vmode}_"
                #     f"{_r1_field}_"
                #     f"{_tnx}_{_tny}"
                # )

                # _init_cam = (
                #     not ss["_cam_initialized"]
                #     .get(_cam_key, False)
                # )

                if _vmode == "슬라이스":
                    _c1, _c2 = st.columns([1.2, 1])
                    with _c1:
                        _r1_sdir = st.selectbox("슬라이스 방향", ["y", "x", "z"],
                                                key="r1_sdir")
                    with _c2:
                        _r1_stream = st.checkbox("유선 표시", value=False,
                                                 key="r1_stream")
                    # 슬라이스 위치는 Plotly 내장 슬라이더로 제어.
                    # v10 항목2·3: n_frames 11→33(3배) + 형상 주변 세밀 적응 분율.
                    # Streamlit 슬라이더를 쓰면 리런 → Plotly.react() → 카메라 리셋.
                    _fig_r1 = _viz_r1.render_field_plotly(
                        _r1_field, _r1_sdir, 0.5, _r1_stream,
                        tile_nx=_tnx, tile_ny=_tny, init_camera=_init_cam,
                        n_frames=33, stl_opacity=_stl_op)
                    _cap = "💡 드래그: 회전 | 스크롤: 줌 | 차트 하단 슬라이더: 슬라이스 위치"
                elif _vmode == "볼륨(연속)":
                    # v10 항목1: 슬라이스 사이를 보간한 연속 볼륨(go.Volume).
                    # v13 항목13: 투명도 0~100%. 항목14: 100%면 최대색이 범례색과
                    # 일치(opacityscale 로 실제 불투명해짐).
                    _volc_pct = st.slider(
                        "볼륨 투명도 [%]", 0, 100,
                        int(round(float(ss.get("r1_opacity_volc", 0.30)) * 100)),
                        5, key="r1_opacity_volc_pct",
                        help="0%=완전 투명, 100%=완전 불투명(최대색이 범례색과 일치)")
                    _op_volc = _volc_pct / 100.0
                    ss["r1_opacity_volc"] = _op_volc
                    # v13 항목15: XYZ 축별 '대칭' 클리핑(-50%~+50%). 각 축 양방향
                    # 슬라이더로 음/양 양쪽에서 독립·동시 절단. -50~+50 → [0,1] 위치
                    # 로 매핑((v+50)/100). 기본 (-50,+50)=전체 표시.
                    _cc1, _cc2, _cc3 = st.columns(3)
                    with _cc1:
                        _clip_x = st.slider(
                            "X 클리핑 [%]", -50, 50,
                            tuple(ss.get("r1_clip_x", (-50, 50))), 5,
                            key="r1_clip_x",
                            help="양끝을 좁히면 −X·+X 양쪽에서 절단(전체=−50~+50)")
                    with _cc2:
                        _clip_y = st.slider(
                            "Y 클리핑 [%]", -50, 50,
                            tuple(ss.get("r1_clip_y", (-50, 50))), 5,
                            key="r1_clip_y", help="−Y·+Y 양방향 절단")
                    with _cc3:
                        _clip_z = st.slider(
                            "Z 클리핑 [%]", -50, 50,
                            tuple(ss.get("r1_clip_z", (-50, 50))), 5,
                            key="r1_clip_z", help="−Z·+Z 양방향 절단")
                    _clip = tuple(((_lo + 50) / 100.0, (_hi + 50) / 100.0)
                                  for (_lo, _hi) in (_clip_x, _clip_y, _clip_z))
                    _fig_r1 = _viz_r1.render_field_volume(
                        _r1_field, opacity=_op_volc, init_camera=_init_cam,
                        stl_opacity=_stl_op, tile_nx=_tnx, tile_ny=_tny,
                        clip=_clip)
                    _cap = ("💡 드래그: 회전 | 스크롤: 줌 — 연속 볼륨 · XYZ 대칭 "
                            "클리핑(±50%, 파란 평면=절단 위치)으로 내부 단면 확인")
                elif _vmode == "입체":
                    # 항목4: 입체 모드 투명도 — 등치면 모드와 독립된 세션 키 사용.
                    # v13 항목13: 0~100%.
                    _volp = st.slider(
                        "투명도(입체) [%]", 0, 100,
                        int(round(float(ss.get("r1_opacity_vol", 0.55)) * 100)),
                        5, key="r1_opacity_vol_pct",
                        help="0%=완전 투명, 100%=완전 불투명. 입체 모드 전용.")
                    _op_vol = _volp / 100.0
                    ss["r1_opacity_vol"] = _op_vol
                    _fig_r1 = _viz_r1.render_field_3d(
                        _r1_field, tile_nx=_tnx, tile_ny=_tny,
                        init_camera=_init_cam, opacity=_op_vol,
                        stl_opacity=_stl_op)
                    _cap = "💡 드래그: 회전 | 스크롤: 줌 (등치면 3개)"
                else:  # 등치면(스윕)
                    # 수동 슬라이더 체크박스 제거 — Plotly 내장 슬라이더가 수동·자동 모두 담당.
                    # ▶ 재생: 자동 스윕  |  차트 하단 슬라이더: 수동 위치 선택
                    # 항목4: 등치면(스윕) 투명도 — 입체 모드와 독립된 세션 키 사용.
                    # v13 항목13: 0~100%.
                    _isop = st.slider(
                        "투명도(등치면) [%]", 0, 100,
                        int(round(float(ss.get("r1_opacity_iso", 0.55)) * 100)),
                        5, key="r1_opacity_iso_pct",
                        help="0%=완전 투명, 100%=완전 불투명. 등치면 모드 전용.")
                    _op_iso = _isop / 100.0
                    ss["r1_opacity_iso"] = _op_iso
                    _fig_r1 = _viz_r1.render_field_3d(
                        _r1_field, anim="sweep", tile_nx=_tnx, tile_ny=_tny,
                        init_camera=_init_cam, opacity=_op_iso,
                        stl_opacity=_stl_op)
                    _cap = "▶ 재생: 자동 스윕 | 차트 하단 슬라이더: 수동 위치 — 회전·확대 유지됨"

                if _fig_r1:
                    # st.plotly_chart(_fig_r1, use_container_width=True, key="r1_chart")
                    # ss["_cam_init_field"] = True

                    if "_cam_initialized" not in ss:
                        ss["_cam_initialized"] = {}

                    _chart_key = (
                        f"r1_chart_"
                        f"{_vmode}_"
                        f"{_r1_field}"
                    )

                    # v11 항목2: Pan 버튼 제거 — 커서 중심 휠 줌(JS) + 우클릭
                    # 팬(네이티브)로 대체. 버튼식 dragmode 전환이 카메라를
                    # 리셋시키는 plotly gl3d 문제의 원천도 함께 제거된다.
                    _fig_r1.update_layout(modebar=dict(remove=["pan3d"]))

                    st.plotly_chart(
                        _fig_r1,
                        use_container_width=True,
                        key=_chart_key
                    )

                    ss["_cam_initialized"][
                        _chart_key
                    ] = True

                    # _chart_key = (
                    #     f"r1_chart_"
                    #     f"{_vmode}_"
                    #     f"{_r1_field}"
                    # )

                    # st.plotly_chart(
                    #     _fig_r1,
                    #     use_container_width=True,
                    #     key=_chart_key
                    # )

                    # ss["_cam_initialized"][
                    #     _cam_key
                    # ] = True


                    # 모든 3D 모드에 JS 주입 — 카메라(회전·확대) 유지.
                    # cam을 클로저 변수 대신 localStorage에 저장한다.
                    # Streamlit이 슬라이더 변경 시 Plotly.relayout()을 추가 호출해
                    # plotly_relayout이 발동되어 클로저 cam이 덮어쓰여지는 문제를 차단.
                    # Plotly.react()는 plotly_relayout을 발동하지 않으므로 localStorage
                    # 값은 사용자 인터랙션으로만 업데이트된다.
                    import streamlit.components.v1 as _cv1
                    _cv1.html("""<script>
(function(){
  // ── 설계 원칙 ──────────────────────────────────────────────────────────
  // Streamlit 은 슬라이스 축/필드 변경(리런) 시 컴포넌트 iframe 은 재사용(스크립트
  // 미실행)하면서 Plotly 차트 DOM 원소는 교체한다. 따라서 "스크립트 재실행 시 1회
  // attach" 방식은 교체된 새 원소에 핸들러를 못 건다. 해결책: 부모 윈도우에 단일
  // 영속 루프를 두어
  //   1) 현재 scene 차트 원소를 매 틱 추적하고, 원소가 바뀌면 자동 재-attach + 복원
  //   2) 라이브 gl 카메라를 폴링해 "안정(2회 연속 동일)·비애니메이션" 상태만 저장
  //      → plotly_relayout 이 놓치는 2차 회전도 포착, 애니메이션 transient 는 제외
  // 카메라는 W.__camStore(부모 윈도우)에 보관 → 리런·모드전환·축변경에도 지속.
  // P(Plotly)는 iframe 초기 실행 시 로드 전일 수 있어 매번 지연 해석(PL).
  // ───────────────────────────────────────────────────────────────────────
  var W=window.parent;
  function PL(){try{return W.Plotly||window.top.Plotly;}catch(e){return null;}}
  function clone(c){return c?JSON.parse(JSON.stringify(
    {eye:c.eye,center:c.center,up:c.up})):null;}
  function liveCam(gd){
    try{
      var s=gd._fullLayout&&gd._fullLayout.scene;
      if(s&&s._scene&&s._scene.getCamera)return s._scene.getCamera();
      return s?s.camera:null;
    }catch(e){return null;}
  }
  function eq(a,b){
    if(!a||!b)return false;
    var ks=['eye','center','up'],ds=['x','y','z'];
    for(var i=0;i<ks.length;i++){
      if(!a[ks[i]]||!b[ks[i]])return false;
      for(var j=0;j<ds.length;j++){
        if(Math.abs(a[ks[i]][ds[j]]-b[ks[i]][ds[j]])>1e-5)return false;}}
    return true;
  }
  function findGd(){
    var ps=W.document.querySelectorAll('.js-plotly-plot'),i,p;
    for(i=0;i<ps.length;i++){p=ps[i];
      if(p._fullLayout&&p._fullLayout.scene&&p.offsetParent!==null)return p;}
    for(i=0;i<ps.length;i++){p=ps[i];
      if(p._fullLayout&&p._fullLayout.scene)return p;}
    return null;
  }
  // 저장된 사용자 카메라를 gd 에 복원(+복원 동안 폴링 게이트)
  function restore(gd){
    var p=PL();if(!p||!W.__camStore)return;
    gd.__animUntil=Date.now()+500;
    // v11: 프로그램적 relayout 이 유발하는 afterplot 을 무시(연쇄 복원 차단)
    gd.__apGuard=Date.now()+900;
    try{p.relayout(gd,{'scene.camera':W.__camStore})['catch'](function(){});}
    catch(e){}
  }
  // 애니메이션(슬라이더 스텝·▶재생) 이벤트 핸들러 부착 — 매 프레임 카메라 복원
  function attachHandlers(gd){
    if(gd.__camH){try{
      gd.removeListener('plotly_animatingframe',gd.__camH.af);
      gd.removeListener('plotly_animated',gd.__camH.ad);
      gd.removeListener('plotly_sliderchange',gd.__camH.sc);
      gd.removeListener('plotly_buttonclicked',gd.__camH.bc);
      gd.removeListener('plotly_relayout',gd.__camH.rl);
      gd.removeListener('plotly_afterplot',gd.__camH.ap);
    }catch(e){}}
    var gate=function(){gd.__animUntil=Date.now()+700;};
    var af=function(){restore(gd);};
    var ad=function(){restore(gd);gd.__animUntil=Date.now()+300;};
    var sc=function(){gate();restore(gd);};
    // v11 항목1: 모드바 버튼(dragmode 전환 등)이 gl3d 카메라를 기본값으로
    // 리셋하는 문제 — 버튼 클릭 직후 저장 카메라를 재적용해 팬·회전·줌이
    // 서로를 리셋하지 않게 한다. 단 'reset camera' 계열 버튼은 사용자 의도가
    // 리셋이므로 저장소를 비워 초기 뷰로 돌아가게 둔다.
    var bc=function(ev){
      gate();
      var n=(ev&&ev.button&&ev.button.attr)||'';
      if(String(n).indexOf('resetCamera')>=0){W.__camStore=null;return;}
      W.setTimeout(function(){restore(gd);},60);
      W.setTimeout(function(){restore(gd);},250);
    };
    // v11 항목1: 사용자 드래그(회전·팬) 종료 시 plotly 가 scene.camera 를
    // relayout 으로 확정 → 그 즉시 저장(폴링 '2회 안정' 대기의 빈틈 제거).
    var rl=function(ev){
      try{
        if(ev&&ev['scene.camera']){
          W.__camStore=clone(ev['scene.camera']);
          gd.__apGuard=Date.now()+400;   // 방금의 사용자 조작 — 복원 불필요
        }
      }catch(e){}
    };
    // Streamlit 리런(Plotly.react)이 uirevision 보존에 실패(gl3d flaky)해도
    // afterplot 직후 저장 카메라로 복원. 단 restore()/사용자 드래그가 유발한
    // afterplot 은 __apGuard 로 무시 — 복원 연쇄(스톰) 방지.
    var ap=function(){
      var now=Date.now();
      if(now<(gd.__apGuard||0))return;
      if(!W.__camStore)return;
      gd.__apGuard=now+900;
      gate();
      W.setTimeout(function(){restore(gd);},30);
    };
    gd.__camH={af:af,ad:ad,sc:sc,bc:bc,rl:rl,ap:ap};
    gd.on('plotly_animatingframe',af);
    gd.on('plotly_animated',ad);
    gd.on('plotly_sliderchange',sc);
    gd.on('plotly_buttonclicked',bc);
    gd.on('plotly_relayout',rl);
    gd.on('plotly_afterplot',ap);
  }

  // ── v11 항목2: Rhino 식 커서 중심 휠 줌 ────────────────────────────────
  // 기본 plotly 휠 줌(화면 중심 기준)을 가로채, 커서 아래 지점이 화면에
  // 고정되도록 eye 와 center 를 함께 이동한다. 회전·팬과 같은 카메라 변환에
  // 누적되며 즉시 __camStore 에 저장된다.
  function attachWheel(gd){
    if(gd.__wheelH){try{
      gd.removeEventListener('wheel',gd.__wheelH,{capture:true});}catch(e){}}
    var V={sub:function(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];},
           add:function(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]];},
           mul:function(a,s){return [a[0]*s,a[1]*s,a[2]*s];},
           crs:function(a,b){return [a[1]*b[2]-a[2]*b[1],
                                     a[2]*b[0]-a[0]*b[2],
                                     a[0]*b[1]-a[1]*b[0]];},
           len:function(a){return Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);},
           nrm:function(a){var l=V.len(a)||1;return [a[0]/l,a[1]/l,a[2]/l];}};
    var pend=null,raf=false;
    function apply(){
      raf=false;
      if(!pend)return;
      var p=PL();
      var s=gd._fullLayout&&gd._fullLayout.scene;
      if(!p||!s||!s._scene){pend=null;return;}
      var cam=s._scene.getCamera();
      var E=[cam.eye.x,cam.eye.y,cam.eye.z];
      var C=[cam.center.x,cam.center.y,cam.center.z];
      var U=[cam.up.x,cam.up.y,cam.up.z];
      var f=pend.f,nx=pend.nx,ny=pend.ny,ar=pend.ar;pend=null;
      var D=V.sub(E,C),dist=V.len(D);
      if(dist<1e-9)return;
      var fwd=V.nrm(V.mul(D,-1));
      var right=V.nrm(V.crs(fwd,U));
      var upv=V.crs(right,fwd);
      // 커서 방향의 중심평면상 목표점 T (fov 45° 가정 — 검증으로 보정치 확인)
      var t=Math.tan(22.5*Math.PI/180);
      var T=V.add(C,V.add(V.mul(right,nx*t*dist*ar),V.mul(upv,ny*t*dist)));
      var C2=V.add(T,V.mul(V.sub(C,T),f));
      var E2=V.add(C2,V.mul(D,f));
      var nc={eye:{x:E2[0],y:E2[1],z:E2[2]},
              center:{x:C2[0],y:C2[1],z:C2[2]},up:cam.up};
      W.__camStore=clone(nc);
      gd.__animUntil=Date.now()+250;   // 폴링이 중간값을 저장하지 않게
      try{p.relayout(gd,{'scene.camera':nc})['catch'](function(){});}catch(e){}
    }
    var h=function(ev){
      var s=gd._fullLayout&&gd._fullLayout.scene;
      if(!s||!s._scene)return;
      ev.preventDefault();ev.stopImmediatePropagation();
      var cv=gd.querySelector('canvas');
      var r=(cv||gd).getBoundingClientRect();
      if(r.width<2||r.height<2)return;
      var nx=((ev.clientX-r.left)/r.width)*2-1;
      var ny=-(((ev.clientY-r.top)/r.height)*2-1);
      var step=Math.exp((ev.deltaY>0?1:-1)*0.14);   // 아래로=축소(f>1)
      if(pend){pend.f*=step;pend.nx=nx;pend.ny=ny;}
      else{pend={f:step,nx:nx,ny:ny,ar:r.width/r.height};}
      if(!raf){raf=true;W.requestAnimationFrame(apply);}
    };
    gd.addEventListener('wheel',h,{capture:true,passive:false});
    gd.__wheelH=h;
  }

  // 단일 영속 루프(부모 윈도우 타이머 — 0-height iframe 로컬 타이머는 throttle 됨)
  if(W.__camLoop){try{W.clearInterval(W.__camLoop);}catch(e){}}
  W.__camPrevPoll=null;
  W.__camLoop=W.setInterval(function(){
    var gd=findGd();if(!gd)return;
    if(gd!==W.__camGd){
      // 원소 교체(리런/모드전환/축변경) → 재-attach + 저장 카메라 복원
      W.__camGd=gd;W.__camPrevPoll=null;
      attachHandlers(gd);
      attachWheel(gd);
      restore(gd);
      // 늦은 Plotly.react 기본값 덮어쓰기 대비 재복원(부모 타이머 — throttle 회피)
      W.setTimeout(function(){restore(gd);},150);
      W.setTimeout(function(){restore(gd);},400);
      return;
    }
    // 카메라 폴링: 안정(2회 연속 동일)·비애니메이션 상태만 저장
    var now=Date.now();
    if(now<(gd.__animUntil||0)){W.__camPrevPoll=null;return;}
    var c=liveCam(gd);if(!c)return;
    if(W.__camPrevPoll&&eq(W.__camPrevPoll,c)){W.__camStore=clone(c);}
    W.__camPrevPoll=clone(c);
  },100);
})();
</script>""", height=0)

                    st.caption(_cap)
                else:
                    st.info("유동장 데이터를 불러오는 중이거나 렌더러를 사용할 수 없습니다.")

        with r_tab2:
            # ── 수렴 이력 (Plotly — 한국어 폰트 불필요) ──────────────────
            if selected_case_dir is None:
                st.info("매트릭스에서 완료/진행 중(☑/⏳) 조건을 선택하세요.")
            else:
                _viz_r2   = CFDVisualizer(selected_case_dir)
                _fig_resid = _viz_r2.plot_residuals_plotly()
                if _fig_resid:
                    # v13 항목10: 다운로드 이미지도 2배 해상도(고DPI 화면 대비)
                    st.plotly_chart(
                        _fig_resid, use_container_width=True, key="r2_resid",
                        config={"toImageButtonOptions": {"scale": 2}})
                    st.caption("hover로 각 반복에서의 잔차 값 확인 가능")
                else:
                    _logs = list(selected_case_dir.glob("*.log")) + list(selected_case_dir.glob("log.*"))
                    if _logs:
                        st.warning(f"수렴 이력 파싱 실패 — 로그 파일 존재: {[l.name for l in _logs[:3]]}")
                    else:
                        st.info("수렴 이력 없음 — 해석 진행 중이거나 로그 파일이 없습니다.")

        with r_tab3:
            # ── 유속 감쇠 (Plotly) ────────────────────────────────────────
            if selected_case_dir is None:
                st.info("매트릭스에서 완료/진행 중(☑/⏳) 조건을 선택하세요.")
            else:
                _viz_r3   = CFDVisualizer(selected_case_dir)
                _fig_atten = _viz_r3.plot_velocity_attenuation_plotly()
                if _fig_atten:
                    st.plotly_chart(_fig_atten, use_container_width=True, key="r3_atten")
                else:
                    st.info("유속 샘플링 데이터 없음 — 전체 구조 모드(full_structure)에서 이용 가능합니다.")

        with r_tab4:
            # ── 항목7·8: 프로젝트(=결과 CSV) 단위 데이터 관리 ──────────────
            # 결과 CSV 한 파일 = 하나의 프로젝트(케이스 묶음). CSV 탭은 '현재 모드'의
            # 프로젝트만 나열하고, 선택한 프로젝트 정보만 표시한다(다른 모드/프로젝트
            # 데이터 혼입 방지). 표시 유속·Cd/Cl 요약은 이 CSV에서 자동 파생되므로
            # 신규 조건이 추가되면 자동 갱신된다.
            import pandas as pd
            # 활성 프로젝트가 있으면 그 폴더의 CSV 만, 없으면 모드 루트의 CSV 를 나열.
            all_csvs = sorted(
                _results_root(mode).glob("*.csv"),
                key=lambda c: c.stat().st_mtime, reverse=True
            )
            # 항목5: 활성 프로젝트가 있으면 그 프로젝트의 결과 CSV 로 고정(타 프로젝트·
            # stale 데이터 혼입 차단). 프로젝트가 없으면 모드 CSV 선택을 허용.
            _active_proj = ss.get("active_project")
            _forced_csv = None
            if _active_proj and ss.get("batch_csv_name"):
                _fp = _results_root(mode) / ss.get("batch_csv_name")
                if _fp.exists():
                    _forced_csv = _fp

            if not all_csvs and _forced_csv is None:
                st.info(f"이 모드({mode})에 저장된 프로젝트(결과 CSV)가 없습니다. "
                        "해석을 완료하면 자동 생성됩니다.")
            elif _active_proj and _forced_csv is None:
                st.info(f"활성 프로젝트 **{_active_proj}** 의 결과 CSV가 아직 없습니다. "
                        "이 프로젝트로 해석을 완료하면 생성됩니다.")
            else:
                if _forced_csv is not None:
                    st.markdown(
                        f"📁 **활성 프로젝트:** {_active_proj} — "
                        f"`{_forced_csv.name}`  (이 프로젝트 데이터만 표시)")
                    csv_target = _forced_csv
                else:
                    _csv_names = [c.name for c in all_csvs]
                    csv_sel = st.selectbox(
                        "📂 프로젝트 불러오기 (결과 CSV 파일)",
                        options=_csv_names,
                        key="result_csv_sel",
                        help="결과 CSV 한 파일이 하나의 프로젝트입니다. 선택한 "
                             "프로젝트의 조건·Cd/Cl만 아래에 표시됩니다.")
                    csv_target = next(c for c in all_csvs if c.name == csv_sel)

                try:
                    df = pd.read_csv(csv_target)

                    # ── 프로젝트 요약 (조건 수·유속·영각) ──────────────────
                    if {"speed_m_s", "angle_deg"}.issubset(df.columns):
                        _p_sp = sorted(df["speed_m_s"].dropna().unique())
                        _p_ag = sorted(df["angle_deg"].dropna().unique())
                        _p_cd = int(df["Cd"].notna().sum()) if "Cd" in df.columns else 0
                        st.caption(
                            f"📁 프로젝트 **{csv_target.stem}** · 조건 {len(df)}개 "
                            f"· 유속 {', '.join(f'{s:.2f}' for s in _p_sp)} m/s "
                            f"· 영각 {', '.join(f'{a:.1f}' for a in _p_ag)}° "
                            f"· Cd/Cl 유효 {_p_cd}행")

                    # ── 데이터 테이블 ──────────────────────────────────────
                    # 주의: CSV 의 Fx/Fy/Fz 는 '솔버 프레임'(유속 +X 고정)이다.
                    # 유동장·미리보기는 '통일 표시 프레임'(유속이 X–Y 평면에서 회전)
                    # 으로 그려지므로 축 이름이 서로 다르다. 같은 화면에서 비교할 수
                    # 있도록 표시 프레임 성분(F_항력/F_양력/F_측력)을 함께 계산해 붙인다.
                    _df_show = df.copy()
                    if {"Fx_N", "Fy_N", "Fz_N", "angle_deg"}.issubset(df.columns):
                        try:
                            from cfd_manager import (solver_to_display_rotation,
                                                     apply_rotation)
                            _Rs = [solver_to_display_rotation(mode, float(r.angle_deg))
                                   for r in df.itertuples()]
                            _fd = [apply_rotation(_R, (float(r.Fx_N), float(r.Fy_N),
                                                       float(r.Fz_N)))
                                   for _R, r in zip(_Rs, df.itertuples())]
                            # 표시 프레임의 힘 성분 — 화면 축과 그대로 대응한다.
                            _df_show["FX_표시 [N]"] = [f[0] for f in _fd]
                            _df_show["FY_표시 [N]"] = [f[1] for f in _fd]
                            _df_show["FZ_표시 [N]"] = [f[2] for f in _fd]
                            # 항력축(=유동방향)은 표시 프레임에서 α 에 따라 X–Y 평면을
                            # 회전한다(α=0 → −Y, α=90 → +X). 고정 라벨을 붙이면 틀리므로
                            # 각 행의 실제 방향을 함께 적는다.
                            _df_show["유동방향_표시"] = [
                                "({:+.2f}, {:+.2f}, {:+.2f})".format(
                                    *apply_rotation(_R, (1.0, 0.0, 0.0)))
                                for _R in _Rs]
                        except Exception:
                            pass

                    # 힘 열은 값이 숫자만 나오므로 단위를 '열 제목'에만 붙인다.
                    # (CSV 파일 자체는 하위 C++ 모델 호환을 위해 원래 열명을 유지 —
                    #  여기서 바꾸는 건 화면 표시용 _df_show 뿐이다.)
                    _df_show = _df_show.rename(columns={
                        "Fx_N": "Fx [N]", "Fy_N": "Fy [N]", "Fz_N": "Fz [N]",
                    })

                    st.markdown(f"**{csv_target.name}** — {len(df)} 행, {len(df.columns)} 열")
                    st.dataframe(
                        _df_show.style.format({
                            col: "{:.5f}" for col in _df_show.select_dtypes("float").columns
                        }),
                        use_container_width=True,
                        height=250,
                    )
                    if "FX_표시 [N]" in _df_show.columns:
                        st.caption(
                            "⚠️ **Fx/Fy/Fz [N] 은 솔버 프레임**(유속을 항상 +X 로 고정)"
                            " 이라 유동장·미리보기 화면의 축 이름과 다릅니다. "
                            "화면과 같은 축의 성분이 **FX·FY·FZ_표시 [N]** 입니다.\n\n"
                            "· **항력 = Fx [N]**(유동방향 성분), **양력 = Fz [N]**(유동에 수직) "
                            "— 이 둘은 프레임과 무관한 값이라 Cd·Cl 과 항상 대응합니다.\n"
                            "· 화면에서 유동방향은 영각에 따라 X–Y 평면을 회전합니다"
                            "(α=0° → −Y축, α=90° → +X축). 행별 실제 방향은 "
                            "**유동방향_표시** 열을 보세요.\n"
                            "· 다운로드되는 CSV 원본 열명(`Fx_N` 등)은 하위 C++ 모델 "
                            "호환을 위해 그대로 유지됩니다."
                        )

                    # ── 영각 / 유속 필터 ──────────────────────────────────
                    _has_filter_cols = "angle_deg" in df.columns and "speed_m_s" in df.columns
                    if _has_filter_cols:
                        _speeds_all  = sorted(df["speed_m_s"].dropna().unique())
                        _angles_all  = sorted(df["angle_deg"].dropna().unique())
                        _sp_opts = [f"{s:.2f}" for s in _speeds_all]
                        _ag_opts = [f"{a:.1f}" for a in _angles_all]
                        # 항목5: 고정 키를 쓰면 세션 선택이 남아 새 유속/영각이 추가돼도
                        # 옛 선택(예: 1.00만)에 가려져 표시되지 않는다. 위젯 키에 'CSV +
                        # 가용 옵션 집합' 시그니처를 포함해, 조건이 바뀌면 위젯이 재생성되며
                        # 항상 전체(default=모든 조건)가 선택되도록 한다.
                        _sig = (csv_target.name + "|" + ",".join(_sp_opts)
                                + "|" + ",".join(_ag_opts))
                        import hashlib as _hl
                        _sig = _hl.md5(_sig.encode()).hexdigest()[:8]

                        _fc1, _fc2 = st.columns(2)
                        with _fc1:
                            _sel_speeds = st.multiselect(
                                "표시할 유속 [m/s]",
                                options=_sp_opts, default=_sp_opts,
                                key=f"r4_speed_filter_{_sig}",
                            )
                        with _fc2:
                            _sel_angles = st.multiselect(
                                "표시할 영각 [°]",
                                options=_ag_opts, default=_ag_opts,
                                key=f"r4_angle_filter_{_sig}",
                            )
                        # 방어: 빈 선택이면 전체로 간주(모든 조건 표시 보장)
                        if not _sel_speeds: _sel_speeds = _sp_opts
                        if not _sel_angles: _sel_angles = _ag_opts

                        # 필터 적용된 임시 CSV 생성
                        _sel_s_vals = [float(s) for s in _sel_speeds]
                        _sel_a_vals = [float(a) for a in _sel_angles]
                        _df_filt = df[
                            df["speed_m_s"].isin(_sel_s_vals) &
                            df["angle_deg"].isin(_sel_a_vals)
                        ]

                        if _df_filt.empty:
                            st.info("선택된 유속/영각 조합의 데이터가 없습니다.")
                        else:
                            import tempfile as _tmp
                            _tmp_csv = Path(_tmp.mktemp(suffix=".csv"))
                            _df_filt.to_csv(_tmp_csv, index=False)
                            _viz_r4 = CFDVisualizer(selected_case_dir or csv_target.parent)
                            _fig_coeff = _viz_r4.plot_force_coefficients_plotly(_tmp_csv)
                            _tmp_csv.unlink(missing_ok=True)
                            if _fig_coeff and len(_fig_coeff.data) > 0:
                                st.plotly_chart(_fig_coeff, use_container_width=True,
                                                key="r4_coeff")
                            else:
                                _valid_cd = _df_filt["Cd"].notna().sum() if "Cd" in _df_filt.columns else 0
                                st.info(f"유효한 Cd/Cl 데이터: {_valid_cd}행 — "
                                        "해석이 완료된 케이스가 없거나 데이터가 NaN입니다.")

                            # ── Cd/Cl 요약 피벗 테이블 (배치 결과) — 항목 6 ──
                            _dfv = _df_filt.dropna(subset=["Cd", "Cl"]) \
                                if {"Cd", "Cl"}.issubset(_df_filt.columns) else _df_filt.iloc[0:0]
                            if not _dfv.empty:
                                with st.expander("📋 Cd / Cl 요약 테이블", expanded=True):
                                    _pt1, _pt2 = st.tabs(["Cd 테이블", "Cl 테이블"])
                                    with _pt1:
                                        _pc = _dfv.pivot_table(values="Cd",
                                                index="angle_deg", columns="speed_m_s")
                                        _pc.index.name = "영각 [°]"
                                        _pc.columns = [f"U={s:.3g}" for s in _pc.columns]
                                        st.dataframe(_pc.style.format("{:.5f}"),
                                                     use_container_width=True)
                                    with _pt2:
                                        _pl = _dfv.pivot_table(values="Cl",
                                                index="angle_deg", columns="speed_m_s")
                                        _pl.index.name = "영각 [°]"
                                        _pl.columns = [f"U={s:.3g}" for s in _pl.columns]
                                        st.dataframe(_pl.style.format("{:.5f}"),
                                                     use_container_width=True)
                    else:
                        _viz_r4 = CFDVisualizer(selected_case_dir or csv_target.parent)
                        _fig_coeff = _viz_r4.plot_force_coefficients_plotly(csv_target)
                        if _fig_coeff and len(_fig_coeff.data) > 0:
                            st.plotly_chart(_fig_coeff, use_container_width=True,
                                            key="r4_coeff")
                        else:
                            st.warning(f"CSV 컬럼: {list(df.columns)}  — "
                                       "angle_deg, Cd, Cl 컬럼이 없습니다.")

                    # ── 다운로드 버튼 ─────────────────────────────────────
                    with open(csv_target, "rb") as _f:
                        st.download_button(
                            "⬇️ CSV 다운로드 (질량-스프링 모델 호환 포맷)",
                            data=_f.read(),
                            file_name=csv_target.name,
                            mime="text/csv",
                            key="r4_download",
                        )
                    # 단위 포함본 — 화면 표에 붙인 단위를 파일에도 그대로 반영한다.
                    # (위 호환 포맷은 하위 C++ 모델이 열명을 그대로 파싱하므로 건드리지
                    #  않고, 사람이 읽는 용도로 별도 파일을 제공한다.)
                    _UNIT_COLS = {
                        "speed_m_s": "speed [m/s]", "angle_deg": "angle [deg]",
                        "Fx_N": "Fx [N]", "Fy_N": "Fy [N]", "Fz_N": "Fz [N]",
                        "rho_kg_m3": "rho [kg/m3]",
                    }
                    _df_unit = _df_show.rename(columns={
                        k: v for k, v in _UNIT_COLS.items() if k in _df_show.columns})
                    st.download_button(
                        "⬇️ CSV 다운로드 (단위 표기 포함 · 사람이 읽는 용)",
                        data=_df_unit.to_csv(index=False).encode("utf-8-sig"),
                        file_name=csv_target.stem + "_units.csv",
                        mime="text/csv",
                        key="r4_download_units",
                        help="열 제목에 단위를 붙이고 표시 프레임 힘 성분까지 포함합니다. "
                             "Cd·Cl·Cm 은 무차원이라 단위가 없습니다.",
                    )

                except Exception as e:
                    st.error(f"CSV 읽기 오류: {e}")
                    st.code(str(csv_target))


# ═══════════════════════════════════════════════════════════════════════════
#  탭 5: 도움말
# ═══════════════════════════════════════════════════════════════════════════

with tab_help:
    st.markdown("""
### 📖 시스템 사용 가이드

이 시스템은 **입력 설정 · 결과 분석 · 도움말** 3개 탭으로 구성됩니다.
단일 해석과 배치 해석은 하나의 인터페이스로 통합되었습니다 — **유속·영각 단계 수를
각각 1로 두면 단일 해석**, 2 이상이면 모든 조합을 순환하는 배치 해석입니다.

---

#### 1️⃣ 입력 설정 탭
1. **STL 업로드**: 단위 셀 모드는 그물 단위 셀 STL 1개, 전체 구조 모드는 가두리 림 + 그물 STL.
2. **계산량 프리셋**: 최소/보통/정밀/최고정밀 중 선택(예상 소요시간 표시).
3. **유속·영각 범위 + 단계 수**: 단계 수 1×1 = 단일 해석, N×M = 배치 해석.
4. **격자·형상 파라미터**: 셀 크기·고형률·정밀화 레벨·반복 횟수.
5. **주기 반복 수 Nx×Ny**(단위 셀): 격자는 항상 1셀(주기 BC)이라 Cd·계산시간 불변 —
   **총 힘[N] 환산**과 **유동장 타일 시각화·투영면적 표시**에만 사용됩니다.
6. **해석 실행**: '해석 시작'(단일) 또는 '배치 해석 시작' 버튼.

**주기 경계조건(Cyclic)**: xMin↔xMax, yMin↔yMax 면이 자동으로 주기 조건이 됩니다.

---

#### 2️⃣ 결과 분석 탭
- **🧮 해석 매트릭스**: 유속×영각 격자에서 셀(✅완료 ⏳진행중 🟡부분 ⚠️중단)을
  **클릭하면 그 조건의 유동장**이 표시됩니다. 미해석 셀은 '해석 시작 전' 안내가 나옵니다.
- **🌊 유동장**: 3가지 표시 방식
  - **슬라이스**: 단면 + 위치 슬라이더 (단위 셀은 Nx×Ny 타일로 복제 표시).
  - **입체**: 등치면 정적 3D — 마우스로 직접 회전·확대.
  - **등치면(스윕)**: 등치값을 자동으로 훑는 애니메이션(▶ 재생) 또는 수동 고정.
  - 회전/이동/줌으로 맞춘 **카메라 앵글·확대는 슬라이더·필드·표시 방식 변경 후에도 유지**됩니다.
- **📉 수렴 이력**: 잔차(Ux/Uy/Uz/p/k/ω) 수렴 그래프.
- **📊 유속 감쇠**: 전체 구조 모드의 가두리 전/후 유속 프로파일.
- **💾 CSV 데이터**: 결과 표 + Cd/Cl vs 영각 통합 그래프 + 요약 피벗 테이블 + 다운로드.

> 정상상태 RANS 해석이라 애니메이션(등치값 스윕)은 시간 변화가 아니라 **등치값 변화**입니다.

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
| Fx_N, Fy_N, Fz_N | 각 방향 힘 [N] (forces 함수 객체로 산출) |
| rho_kg_m3 | 해수 밀도 [kg/m³] |

케이스 디렉토리는 실행 중 `해석중_`, 완료 후 `해석완료_` 접두어로 구분됩니다.

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

# ─── 자동 새로고침 (모든 탭 렌더링 이후, 스크립트 맨 끝) ─────────────────────
# 해석 실행 중일 때만 갱신 주기마다 재실행한다. 반드시 모든 탭이 렌더링된 뒤에
# 호출해야 결과분석·도움말 탭이 정상 표시된다. (이전엔 모니터 탭 안에서 호출해
# 뒤 탭들이 빈 화면이 됐음)
if ss.job_status == "running":
    time.sleep(ss.get("refresh_interval", 3))
    st.rerun()
elif _status_at_render == "running" and ss.job_status in ("done", "error"):
    # 이번 실행은 'running'(예: 96.8%)으로 화면을 그렸는데, 렌더 도중 백그라운드
    # 스레드가 done/error로 완료했다. 위 조건은 이미 False라 재실행이 일어나지 않아
    # 화면이 직전 진행률에 멈춘다. 최종 상태(100%·완료)를 칠하기 위해 한 번 더 그린다.
    st.rerun()
