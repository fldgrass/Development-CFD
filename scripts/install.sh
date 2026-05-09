#!/usr/bin/env bash
# =============================================================================
# install.sh
# 양식 가두리 CFD 해석 시스템 통합 설치 스크립트
# Ubuntu 20.04 / 22.04 LTS 기준
# NVIDIA RTX 3090 × 2 환경
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${BLUE}━━━ Step $* ━━━${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$BASE_DIR/logs/install_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$BASE_DIR/logs"

# ─── 배너 ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}${BLUE}"
cat << 'BANNER'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    양식 가두리 CFD 해석 시스템 — 설치 스크립트               ║
║    OpenFOAM v2312 + Python + Streamlit + PyVista             ║
║    NVIDIA RTX 3090 × 2 최적화 환경                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

log_info "설치 로그: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# ─── OS 확인 ─────────────────────────────────────────────────────────────
log_step "1 — 시스템 요구사항 확인"

OS_ID=$(. /etc/os-release && echo "$ID")
OS_VER=$(. /etc/os-release && echo "$VERSION_ID")
log_info "운영체제: $OS_ID $OS_VER"

if [[ "$OS_ID" != "ubuntu" ]]; then
    log_warn "Ubuntu 이외의 OS입니다. 일부 단계가 건너뛰어질 수 있습니다."
fi

# CPU 코어 수
CPU_CORES=$(nproc --all)
log_info "CPU 코어: ${CPU_CORES}개"

# NVIDIA GPU 확인
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    log_ok "NVIDIA GPU ${GPU_COUNT}개 감지됨:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | while IFS= read -r line; do log_info "  $line"; done
else
    log_warn "NVIDIA GPU 미감지 — CPU 모드로 설치됩니다."
fi

# RAM 확인
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
log_info "RAM: ${RAM_GB} GB"
if [ "$RAM_GB" -lt 16 ]; then
    log_warn "RAM이 16GB 미만입니다. 대용량 격자 생성 시 부족할 수 있습니다."
fi

# ─── 시스템 패키지 업데이트 ──────────────────────────────────────────────
log_step "2 — 시스템 패키지 업데이트"
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget curl \
    python3 python3-pip python3-venv python3-dev \
    openmpi-bin openmpi-common libopenmpi-dev \
    libgl1-mesa-glx libgl1-mesa-dev \
    libglu1-mesa libglu1-mesa-dev \
    xvfb \
    paraview \
    software-properties-common \
    gnome-desktop-manager \
    2>/dev/null || log_warn "일부 패키지 설치 건너뜀"

log_ok "시스템 패키지 설치 완료"

# ─── OpenFOAM 설치 ───────────────────────────────────────────────────────
log_step "3 — OpenFOAM v2312 설치"

if command -v blockMesh &>/dev/null; then
    log_ok "OpenFOAM이 이미 설치되어 있습니다."
else
    log_info "OpenFOAM 저장소 추가 중..."

    # ESI OpenFOAM 저장소 (openfoam.com)
    curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash 2>/dev/null || {
        log_warn "저장소 자동 추가 실패 — 수동 설치를 진행합니다."
        log_info "수동 설치 명령:"
        log_info "  wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash"
        log_info "  sudo apt-get install openfoam2312"
    }

    if sudo apt-cache show openfoam2312 &>/dev/null; then
        sudo apt-get install -y openfoam2312
        log_ok "OpenFOAM v2312 설치 완료"
    elif sudo apt-cache show openfoam &>/dev/null; then
        sudo apt-get install -y openfoam
        log_ok "OpenFOAM 설치 완료 (최신 버전)"
    else
        log_warn "OpenFOAM 자동 설치 실패."
        log_warn "수동 설치: https://openfoam.com/download/"
        log_warn "또는: sudo apt-get install openfoam2306"
    fi

    # bashrc 자동 추가
    for RC_FILE in "$HOME/.bashrc" "$HOME/.bash_profile"; do
        if [ -f "$RC_FILE" ] && ! grep -q "openfoam" "$RC_FILE"; then
            for OF_BASHRC in /usr/lib/openfoam/openfoam2312/etc/bashrc \
                             /opt/openfoam*/etc/bashrc; do
                if [ -f "$OF_BASHRC" ]; then
                    echo "" >> "$RC_FILE"
                    echo "# OpenFOAM 환경" >> "$RC_FILE"
                    echo "source $OF_BASHRC" >> "$RC_FILE"
                    log_ok "OpenFOAM 환경 추가됨: $RC_FILE"
                    break
                fi
            done
        fi
    done
fi

# MPI 확인
if command -v mpirun &>/dev/null; then
    MPI_VER=$(mpirun --version 2>&1 | head -1)
    log_ok "MPI: $MPI_VER"
else
    log_warn "MPI 미감지. 병렬 해석 불가."
fi

# ─── CUDA / AmgX 설치 ────────────────────────────────────────────────────
log_step "4 — CUDA 및 AmgX GPU 가속 설치"

if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    log_ok "CUDA: $CUDA_VER (이미 설치됨)"
else
    log_info "CUDA Toolkit 설치 시도 중..."
    # CUDA 12.x 설치 (Ubuntu 22.04 기준)
    CUDA_PKG="cuda-toolkit-12-2"
    if ! sudo apt-get install -y "$CUDA_PKG" 2>/dev/null; then
        log_warn "CUDA 자동 설치 실패."
        log_info "수동 설치: https://developer.nvidia.com/cuda-downloads"
        log_info "RTX 3090: CUDA 11.8 이상 필요"
    fi
fi

# AmgX (OpenFOAM GPU 가속 플러그인)
AMGX_DIR_DEFAULT="$HOME/AmgX"
if [ -d "$AMGX_DIR_DEFAULT" ]; then
    log_ok "AmgX 이미 설치됨: $AMGX_DIR_DEFAULT"
else
    log_info "AmgX 빌드 방법 (옵션, CUDA 설치 후):"
    log_info "  git clone https://github.com/NVIDIA/AMGX.git ~/AmgX"
    log_info "  cd ~/AmgX && mkdir build && cd build"
    log_info "  cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86"
    log_info "  (RTX 3090: Compute Capability 8.6 → -DCUDA_ARCH=86)"
    log_info "  make -j$(nproc)"
    log_warn "AmgX 미설치 — CPU GAMG 솔버를 사용합니다."
fi

# ─── Python 환경 설치 ────────────────────────────────────────────────────
log_step "5 — Python 가상환경 및 패키지 설치"

VENV_DIR="$BASE_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    log_ok "가상환경 생성: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

log_info "Python 패키지 설치 중 (시간이 걸릴 수 있습니다)..."
pip install -q \
    streamlit>=1.35.0 \
    pyvista[all]>=0.43.0 \
    vtk>=9.3.0 \
    numpy>=1.26.0 \
    pandas>=2.1.0 \
    matplotlib>=3.8.0 \
    scipy>=1.11.0 \
    plotly>=5.18.0 \
    tqdm>=4.66.0 \
    ipywidgets \
    stpyvista \
    2>/dev/null

log_ok "Python 패키지 설치 완료"
pip list | grep -E "streamlit|pyvista|numpy|pandas|matplotlib" | \
    while IFS= read -r line; do log_info "  $line"; done

deactivate

# ─── 권한 설정 ───────────────────────────────────────────────────────────
log_step "6 — 실행 권한 설정"

chmod +x "$SCRIPT_DIR"/*.sh
chmod +x "$SCRIPT_DIR"/*.py 2>/dev/null || true

# 결과 디렉토리 생성
mkdir -p "$BASE_DIR/results/unit_cell"
mkdir -p "$BASE_DIR/results/full_structure"
mkdir -p "$BASE_DIR/stl_uploads"
mkdir -p "$BASE_DIR/logs"

log_ok "디렉토리 구조 및 권한 설정 완료"

# ─── 바탕화면 아이콘 생성 ────────────────────────────────────────────────
log_step "7 — 바탕화면 아이콘 생성"

DESKTOP_DIR="$HOME/Desktop"
mkdir -p "$DESKTOP_DIR"

# 메인 대시보드 아이콘
cat > "$DESKTOP_DIR/CFD_Dashboard.desktop" << DESKEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=🌊 가두리 CFD 대시보드
Comment=양식 가두리 OpenFOAM CFD 해석 시스템
Exec=bash -c 'cd "$BASE_DIR" && source .venv/bin/activate && streamlit run app.py --server.port 8501 --server.headless false & sleep 3 && xdg-open http://localhost:8501'
Icon=$BASE_DIR/assets/cfd_icon.png
Terminal=true
Categories=Science;Engineering;
StartupNotify=true
DESKEOF
chmod +x "$DESKTOP_DIR/CFD_Dashboard.desktop"

# 단위 셀 모드 빠른 실행 아이콘
cat > "$DESKTOP_DIR/CFD_UnitCell.desktop" << DESKEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=🔬 단위 셀 해석
Comment=그물 단위 셀 CFD 해석 (Cd/Cl DB 추출)
Exec=bash -c 'cd "$BASE_DIR" && bash scripts/run_unit_cell.sh; read -p "완료. Enter를 누르세요."'
Terminal=true
Categories=Science;Engineering;
DESKEOF
chmod +x "$DESKTOP_DIR/CFD_UnitCell.desktop"

# 전체 구조 모드 아이콘
cat > "$DESKTOP_DIR/CFD_FullStructure.desktop" << DESKEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=🏗️ 전체 구조 해석
Comment=원통형 가두리 전체 유동장 해석
Exec=bash -c 'cd "$BASE_DIR" && bash scripts/run_full_structure.sh; read -p "완료. Enter를 누르세요."'
Terminal=true
Categories=Science;Engineering;
DESKEOF
chmod +x "$DESKTOP_DIR/CFD_FullStructure.desktop"

# 배치 해석 아이콘
cat > "$DESKTOP_DIR/CFD_Batch.desktop" << DESKEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=🔄 배치 해석
Comment=영각×유속 자동 배치 해석 및 CSV 생성
Exec=bash -c 'cd "$BASE_DIR" && bash scripts/run_batch.sh; read -p "완료. Enter를 누르세요."'
Terminal=true
Categories=Science;Engineering;
DESKEOF
chmod +x "$DESKTOP_DIR/CFD_Batch.desktop"

log_ok "바탕화면 아이콘 4개 생성 완료"

# ─── 런처 스크립트 생성 ──────────────────────────────────────────────────
log_step "8 — 시스템 런처 스크립트 생성"
bash "$SCRIPT_DIR/create_launcher.sh" 2>/dev/null || true

# ─── 설치 완료 요약 ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
cat << 'DONE'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                  ✅  설치 완료!                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
DONE
echo -e "${NC}"

log_ok "시스템 구성 요약:"
log_info "  • 프로젝트 경로: $BASE_DIR"
log_info "  • Python 환경 : $VENV_DIR"
log_info "  • 대시보드 URL : http://localhost:8501"
log_info ""
log_ok "시작 방법:"
log_info "  방법 1: 바탕화면의 '🌊 가두리 CFD 대시보드' 더블클릭"
log_info "  방법 2: 터미널에서 → bash $BASE_DIR/launch.sh"
log_info "  방법 3: 터미널에서 → source .venv/bin/activate && streamlit run app.py"
