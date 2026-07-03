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

def _persist_input_state():
    """업로드한 STL 선택과 자동 감지 정보를 디스크에 기록한다. STL 파일 자체는
    이미 stl_uploads/ 에 저장돼 있으므로, 여기서는 '어떤 파일을 쓰는지'와 감지
    결과만 저장해 두면 새로고침 후 새 세션이 그대로 복구할 수 있다."""
    # v13 항목1: 활성 프로젝트가 있으면 STL 이 없어도 상태를 기록한다(F5 후 복구).
    if not (ss.get("stl_net_path") or ss.get("stl_cage_path")
            or ss.get("active_project")):
        return
    try:
        state = {k: ss.get(k) for k in (
            "analysis_mode", "stl_net_path", "stl_cage_path",
            "auto_cell_size_mm", "auto_wire_d_mm", "auto_solidity",
            "auto_frontal_area", "cell_size_mm", "solidity_input",
            "unit_nx", "unit_ny",
            # v13 항목1: 활성 프로젝트도 저장 → 새로고침 후 자동 로드
            "active_project",
        )}
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
    for k in ("auto_cell_size_mm", "auto_wire_d_mm", "auto_solidity",
              "auto_frontal_area", "cell_size_mm", "solidity_input",
              "unit_nx", "unit_ny"):
        if data.get(k) is not None:
            ss[k] = data[k]
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
    # 결과 CSV(프로젝트 데이터) + 시각화 설정
    "batch_csv_name",
    "r1_viewmode", "r1_field", "r1_opacity_vol", "r1_opacity_iso",
]
# '새 프로젝트' 시 초기화할 기본값(깨끗한 상태)
PROJECT_DEFAULTS = {
    "u_min": 1.0, "u_max": 1.0, "u_steps": 1,
    "a_min": 0.0, "a_max": 0.0, "a_steps": 1,
    "rho": 1025.0, "nu": 1.19, "ti": 5,
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


def _project_has_unsaved_changes(_mode):
    """현재 세션의 프로젝트 설정이 저장된 project.json 과 다른지(항목5).
    활성 프로젝트가 없으면(임시 작업) 변경 여부를 판단할 기준이 없으므로,
    조건/결과가 하나라도 설정돼 있으면 '미저장'으로 본다."""
    _name = ss.get("active_project")
    if not _name:
        # 임시 작업: 기본값과 다른 조건이 하나라도 있으면 미저장으로 간주
        for k, v in PROJECT_DEFAULTS.items():
            if k in ss and ss.get(k) != v:
                return True
        return bool(ss.get("stl_net_path") or ss.get("stl_cage_path"))
    _p = _project_dir(_mode, _name) / "project.json"
    if not _p.exists():
        return True
    try:
        _saved = json.loads(_p.read_text())
    except Exception:
        return True
    def _norm(v):
        return str(v) if isinstance(v, Path) else v
    for k in PROJECT_KEYS:
        if _norm(ss.get(k)) != _norm(_saved.get(k)):
            return True
    return False


@st.dialog("💾 변경사항을 저장할까요?")
def _unsaved_guard_dialog(_next_action: str):
    """항목5: 현재 프로젝트에 미저장 변경이 있을 때, 파괴적 동작(새 프로젝트·
    다른 프로젝트 불러오기) 직전에 저장 여부를 확인한다."""
    _mode = ss.get("analysis_mode", "unit_cell")
    _act_label = {"new": "새 프로젝트 만들기",
                  "load": "다른 프로젝트 불러오기"}.get(_next_action, "계속")
    st.write(f"현재 프로젝트에 저장하지 않은 변경사항이 있습니다. "
             f"**{_act_label}** 전에 저장할까요?")
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


@st.dialog("🗂️ 프로젝트 열기")
def _open_project_dialog():
    """항목4: Windows 파일 탐색기 스타일 '열기' 대화상자에 준하는 브라우저.
    웹앱(브라우저 샌드박스)에서는 OS 네이티브 파일 탐색기를 띄울 수 없으므로,
    프로젝트 폴더를 나열·미리보기·선택하는 표준 다이얼로그로 대체한다."""
    _mode = ss.get("analysis_mode", "unit_cell")
    _root = _projects_dir(_mode)
    st.caption(f"📁 위치: `{_root}`")
    _names = list_project_names(_mode)
    if not _names:
        st.info("이 모드에 저장된 프로젝트가 없습니다.")
        if st.button("닫기", use_container_width=True):
            st.rerun()
        return
    # 파일 목록(수정시각·조건수 미리보기)
    _rows = []
    for _n in _names:
        _pj = _project_dir(_mode, _n) / "project.json"
        _mt, _cond = "-", "-"
        try:
            _mt = datetime.fromtimestamp(_pj.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            _d = json.loads(_pj.read_text())
            _us = int(_d.get("u_steps", 1) or 1); _as = int(_d.get("a_steps", 1) or 1)
            _cond = f"{_us}×{_as}"
        except Exception:
            pass
        _rows.append({"프로젝트": _n, "수정": _mt, "조건(U×A)": _cond})
    st.dataframe(_rows, use_container_width=True, hide_index=True, height=210)
    _sel = st.selectbox("열 프로젝트 선택", options=_names,
                        key="_open_dlg_sel")
    _c1, _c2 = st.columns(2)
    if _c1.button("📂 열기", type="primary", use_container_width=True):
        # 미저장 변경이 있으면 가드 후 로드, 없으면 즉시 로드
        if _project_has_unsaved_changes(_mode):
            ss["_pending_load_after_guard"] = _sel
            ss["_guard_from_dialog"] = True
        else:
            ss._pending_load_project = _sel
            ss["_last_synced_proj_name"] = None
        st.rerun()
    if _c2.button("취소", use_container_width=True):
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
# 파일 대화상자에서 미저장 변경이 감지된 경우 → 가드 다이얼로그를 띄운다.
if ss.pop("_guard_from_dialog", False):
    _unsaved_guard_dialog("load")

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
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    aLen  = span * 0.55
    ax0, az0 = cx - cos_t * aLen,        cz - sin_t * aLen
    ax1, az1 = cx + cos_t * aLen * 0.25, cz + sin_t * aLen * 0.25

    # ── Figure 조립 ───────────────────────────────────────────────────────
    fig = go.Figure()

    # STL 메쉬
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=ii, j=ji, k=ki,
        color='#5dade2', opacity=0.78, flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.2, roughness=0.5),
        lightposition=dict(x=1, y=2, z=3),
        showlegend=False,
        hoverinfo='skip',
    ))

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
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode='lines',
                line=dict(color='#f39c12', width=1),
                showlegend=False, hoverinfo='skip',
            ))

    # 유속 화살표 shaft
    fig.add_trace(go.Scatter3d(
        x=[ax0, ax1], y=[cy, cy], z=[az0, az1],
        mode='lines',
        line=dict(color='#e74c3c', width=6),
        showlegend=False, hoverinfo='skip',
    ))
    # 유속 화살표 cone
    fig.add_trace(go.Cone(
        x=[ax1], y=[cy], z=[az1],
        u=[cos_t * aLen * 0.22], v=[0.0], w=[sin_t * aLen * 0.22],
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
    _scene = dict(
        xaxis=dict(title="X [mm]", backgroundcolor="#eaf4fb",
                   gridcolor="white", showbackground=True),
        yaxis=dict(title="Y [mm]", backgroundcolor="#eaf4fb",
                   gridcolor="white", showbackground=True),
        zaxis=dict(title="Z [mm]", backgroundcolor="#dce9f5",
                   gridcolor="white", showbackground=True),
        aspectmode='data',
        bgcolor='rgba(240,248,255,1)',
    )
    if init_camera:
        _scene['camera'] = dict(eye=dict(x=1.4, y=1.0, z=0.9))

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
        margin=dict(l=0, r=0, t=10, b=0),
        height=440,
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
        st.button("💾 현재 프로젝트 저장", use_container_width=True,
                  key="proj_save_btn", on_click=_do_save_project)
        if ss.pop("_save_toast", None):
            st.success(f"프로젝트 저장됨 · 결과/CSV/매트릭스 유지")
        if ss.pop("_save_toast_warn", None):
            st.warning("프로젝트 이름을 입력하세요.")

        _projs = list_project_names(mode)
        if _projs:
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
            st.button("📂 프로젝트 불러오기", use_container_width=True,
                      key="proj_load_btn", on_click=_do_load_project)
            # v13 항목4: '파일 탐색기에서 열기' — 웹앱은 브라우저 샌드박스라
            # OS 네이티브 파일 탐색기를 띄울 수 없으므로, 프로젝트 폴더를
            # 탐색·선택하는 표준 다이얼로그(브라우저)로 제공한다.
            if st.button("🗂️ 파일에서 프로젝트 열기…", use_container_width=True,
                         key="proj_browse_btn"):
                _open_project_dialog()
        else:
            st.caption("저장된 프로젝트가 없습니다.")

        def _do_new_project():
            # v13 항목5: 새 프로젝트 진입 전 미저장 변경 감지 → 있으면 확인
            # 다이얼로그, 없으면 곧바로 새 프로젝트 대화상자.
            ss["_new_proj_requested"] = True
        st.button("🆕 새 프로젝트 (이름 입력)", use_container_width=True,
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
    rho = st.number_input("해수 밀도 ρ [kg/m³]", value=1025.0,
                          min_value=1000.0, max_value=1100.0, step=1.0, key="rho")
    nu  = st.number_input("동점성계수 ν [×10⁻⁶ m²/s]",
                          value=1.19, min_value=0.5, max_value=2.0, step=0.01, key="nu")
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
    }
    add_log(f"계산 조건: 반복 {_bp['end_time']} · 정밀화 {_bp['refine_level']} · "
            f"수렴 {_bp['residual_control']:.0e} · {n_cores}코어")

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
                    min_value=1, max_value=4, step=1,
                    key="refine_level_preset",
                    help="snappyHexMesh 표면 최대 정밀화 레벨 (min = 레벨-1). "
                         "레벨 3: ~50만 셀(권장), 레벨 4: ~200만 셀(정밀)",
                )
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
                f"| Reynolds = {speed_val * cell_size / 1.19e-6:.1f}"
            )

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
            Re = speed_val * cage_d / 1.19e-6
            st.info(f"📐 **가두리 Reynolds 수** = {Re:.2e}  |  도메인: {3*cage_d:.0f}D × {3*cage_d:.0f}D × {cage_h:.0f}m")

    with col_right:
        # ─── STL 미리보기 (인터랙티브 3D) ────────────────────────────────
        _stl_show = ss.get("stl_net_path") or ss.get("stl_cage_path")
        if _stl_show and Path(_stl_show).exists():
            st.markdown("### 🖼️ STL 미리보기")
            _angle_v  = float(angle_val)   # 대표 영각(목록 첫 값)
            _nx_v     = int(ss.get("unit_nx", 1))
            _ny_v     = int(ss.get("unit_ny", 1))
            _cs_m     = float(ss.get("cell_size_mm", 20.0)) / 1000.0

            # ── 인터랙티브 Plotly 뷰어 (마우스 드래그 회전 가능) ──────────
            # 첫 렌더에만 카메라를 지정하고, 이후 영각·nx·ny 변경 렌더에서는 빼서
            # uirevision이 사용자의 마우스 카메라를 보존하게 한다.
            _stl_init_cam = not ss.get("_cam_init_stl", False)
            _fig3d = render_stl_interactive_plotly(
                _stl_show, mode, _angle_v, _nx_v, _ny_v, _cs_m,
                init_camera=_stl_init_cam,
            )
            if _fig3d is not None:
                st.plotly_chart(_fig3d, use_container_width=True,
                                key="stl_3d_preview")
                ss["_cam_init_stl"] = True
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
        _ec1.metric("총 예상 시간", fmt_duration(_est_total))
        _ec2.caption(
            f"케이스 **{_n_cases}개 × 약 {fmt_duration(_est_per)}/케이스** "
            f"(전처리+솔버 포함) = 총 **{fmt_duration(_est_total)}**. "
            f"반복 {int(end_time)} · 정밀화 {int(_rl_e)} · {n_cores}코어. "
            f"실측 보정 반영 · 수렴 먼저 도달 시 더 빨리 끝납니다."
        )
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

        btn_col1, btn_col2 = st.columns(2)

        # 항목9: 실행 전 이미 완료된 조건과의 중복 검사(덮어쓰기 경고용)
        _run_conds = {(round(float(s), 2), round(float(a), 1))
                      for s in speeds for a in angles}
        _overlap_conds = sorted(_run_conds & _scan_done_conditions(mode))

        with btn_col1:
            if st.button(_run_label, disabled=run_disabled,
                         use_container_width=True, type="primary"):
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
        r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
            "🌊 유동장", "📉 수렴 이력", "📊 유속 감쇠", "💾 CSV 데이터"
        ])

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
                    st.markdown(f"**{csv_target.name}** — {len(df)} 행, {len(df.columns)} 열")
                    st.dataframe(
                        df.style.format({
                            col: "{:.5f}" for col in df.select_dtypes("float").columns
                        }),
                        use_container_width=True,
                        height=250,
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
