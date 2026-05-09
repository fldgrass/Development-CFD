#!/usr/bin/env bash
# =============================================================================
# run_unit_cell.sh
# 단위 셀(Unit Cell) 해석 자동 실행 스크립트
# RTX 3090 x2 / MPI 병렬 / GPU AmgX 가속
#
# 사용법:
#   ./run_unit_cell.sh [유속(m/s)] [영각(deg)] [단위셀크기(mm)] [코어수]
# 예시:
#   ./run_unit_cell.sh 1.0 0 20 16
#   ./run_unit_cell.sh 2.0 30 15 8
# =============================================================================

set -euo pipefail

# ─── 색상 출력 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${BLUE}▶ $*${NC}"; }

# ─── 인수 처리 ──────────────────────────────────────────────────────────────
U_SPEED="${1:-1.0}"     # 유속 [m/s]
AOA="${2:-0}"           # 영각 [degree]
CELL_MM="${3:-20}"      # 단위 셀 크기 [mm]
N_CORES="${4:-0}"       # MPI 코어 수 (0=자동)

# 코어 수 자동 감지
if [ "$N_CORES" -eq 0 ] 2>/dev/null; then
    N_CORES=$(nproc --all)
    # 32 코어 초과 방지
    N_CORES=$(( N_CORES > 32 ? 32 : N_CORES ))
fi

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATE_DIR="$BASE_DIR/openfoam/unit_cell"
RESULTS_DIR="$BASE_DIR/results/unit_cell"
LOGS_DIR="$BASE_DIR/logs"
STL_DIR="$BASE_DIR/stl_uploads"

# 케이스 이름 생성
CASE_NAME="unit_cell_U${U_SPEED}_A${AOA}_$(date +%Y%m%d_%H%M%S)"
CASE_DIR="$RESULTS_DIR/$CASE_NAME"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/${CASE_NAME}.log"

# ─── 배너 출력 ──────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   양식 가두리 단위 셀 CFD 해석 시스템                 ║"
echo "║   OpenFOAM + MPI + GPU(AmgX) 자동화 스크립트          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "유속       : ${U_SPEED} m/s"
log_info "영각       : ${AOA}°"
log_info "단위셀크기 : ${CELL_MM} mm"
log_info "MPI 코어   : ${N_CORES}개"
log_info "케이스     : ${CASE_NAME}"
log_info "로그 파일  : ${LOG_FILE}"

# ─── OpenFOAM 환경 로드 ─────────────────────────────────────────────────────
log_step "OpenFOAM 환경 초기화"

OF_BASHRC=""
for candidate in \
    "/opt/openfoam10/etc/bashrc" \
    "/opt/openfoam9/etc/bashrc" \
    "/opt/openfoam8/etc/bashrc" \
    "/usr/lib/openfoam/openfoam2312/etc/bashrc" \
    "/usr/lib/openfoam/openfoam2206/etc/bashrc" \
    "$HOME/OpenFOAM/OpenFOAM-v2312/etc/bashrc" \
    "$HOME/OpenFOAM/OpenFOAM-v2206/etc/bashrc"; do
    if [ -f "$candidate" ]; then
        OF_BASHRC="$candidate"
        break
    fi
done

if [ -n "$OF_BASHRC" ]; then
    # shellcheck disable=SC1090
    source "$OF_BASHRC"
    log_ok "OpenFOAM 로드: $OF_BASHRC"
else
    # PATH에 이미 있는 경우
    if command -v blockMesh &>/dev/null; then
        log_ok "OpenFOAM이 PATH에서 감지됨"
    else
        log_error "OpenFOAM을 찾을 수 없습니다."
        log_error "설치 후 /opt/openfoam*/etc/bashrc 경로를 확인하세요."
        exit 1
    fi
fi

# ─── GPU/CUDA 환경 ───────────────────────────────────────────────────────────
log_step "GPU 환경 확인 (RTX 3090 × 2)"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        echo "$GPU_INFO" | while IFS= read -r line; do
            log_ok "GPU: $line"
        done
        # CUDA 환경 변수
        export CUDA_VISIBLE_DEVICES="0,1"
        log_ok "CUDA_VISIBLE_DEVICES=0,1 (두 GPU 활성화)"
    fi
else
    log_warn "nvidia-smi 미감지 → CPU 모드로 실행"
fi

# AmgX 확인
if [ -n "${AMGX_DIR:-}" ] && [ -d "$AMGX_DIR" ]; then
    log_ok "AmgX GPU 가속기 감지됨: $AMGX_DIR"
else
    log_warn "AmgX 미감지 → CPU GAMG 솔버 사용 (설치 시 GPU 가속 자동 활성화)"
fi

# ─── 케이스 디렉토리 구성 ────────────────────────────────────────────────────
log_step "케이스 디렉토리 구성"

cp -r "$TEMPLATE_DIR" "$CASE_DIR"
log_ok "템플릿 복사 완료: $CASE_DIR"

# STL 파일 복사 (가장 최근 net STL)
TRISURF_DIR="$CASE_DIR/constant/triSurface"
mkdir -p "$TRISURF_DIR"

NET_STL=$(ls -t "$STL_DIR"/net*.stl 2>/dev/null | head -1 || echo "")
if [ -z "$NET_STL" ]; then
    log_warn "STL 파일을 찾을 수 없습니다. $STL_DIR 에 net*.stl 파일을 배치하세요."
    log_warn "데모 모드: 더미 STL 생성 (실제 해석에는 실제 STL 필요)"
    # 더미 구체 STL 생성 (테스트용)
    cat > "$TRISURF_DIR/netSurface.stl" << 'STLEOF'
solid netSurface
  facet normal 0 0 1
    outer loop
      vertex -0.001 -0.001 0.0005
      vertex  0.001 -0.001 0.0005
      vertex  0.000  0.001 0.0005
    endloop
  endfacet
endsolid netSurface
STLEOF
    log_warn "더미 netSurface.stl 생성됨"
else
    cp "$NET_STL" "$TRISURF_DIR/netSurface.stl"
    log_ok "STL 복사: $(basename "$NET_STL")"
fi

# ─── 속도 벡터 계산 (영각 적용) ──────────────────────────────────────────────
log_step "경계 조건 설정 (영각 ${AOA}° 적용)"

# Python으로 속도 벡터 계산
VELOCITY=$(python3 -c "
import math
u = float('$U_SPEED')
aoa = math.radians(float('$AOA'))
ux = u * math.cos(aoa)
uz = u * math.sin(aoa)
print(f'{ux:.6f} 0.000000 {uz:.6f}')
")
UX=$(echo "$VELOCITY" | awk '{print $1}')
UZ=$(echo "$VELOCITY" | awk '{print $3}')

log_info "속도 벡터: ($UX, 0, $UZ) m/s"

# turbulence parameters
TURB_K=$(python3 -c "
u = float('$U_SPEED')
I = 0.05
k = 1.5 * (I*u)**2
print(f'{k:.6e}')
")
TURB_OMEGA=$(python3 -c "
import math
u = float('$U_SPEED')
k = 1.5 * (0.05*u)**2
L = float('$CELL_MM') / 1000.0
omega = math.sqrt(k) / (0.09**0.25 * L)
print(f'{omega:.4f}')
")

log_info "난류 k = $TURB_K m²/s², ω = $TURB_OMEGA 1/s"

# ─── 경계조건 파일 수정 ───────────────────────────────────────────────────────
# U 파일 수정
sed -i "s|uniform (1.0 0 0);|uniform ($UX 0 $UZ);|g" "$CASE_DIR/0/U"
sed -i "s|uniform 3.75e-3;|uniform $TURB_K;|g"       "$CASE_DIR/0/k"
sed -i "s|uniform 3.45;|uniform $TURB_OMEGA;|g"       "$CASE_DIR/0/omega"

# 단위 셀 크기에 맞게 blockMeshDict 수정
CELL_M=$(python3 -c "print(float('$CELL_MM')/1000)")
python3 "$BASE_DIR/scripts/patch_blockMesh_unit.py" \
    "$CASE_DIR/system/blockMeshDict" "$CELL_MM" || true

# ─── decomposeParDict 코어 수 수정 ───────────────────────────────────────────
sed -i "s|numberOfSubdomains  16;|numberOfSubdomains  $N_CORES;|g" \
    "$CASE_DIR/system/decomposeParDict"

# controlDict 기준 유속 수정
sed -i "s|magUInf         1.0;|magUInf         $U_SPEED;|g" \
    "$CASE_DIR/system/controlDict"

log_ok "경계조건 설정 완료"

# ─── surfaceFeatureExtractDict 자동 생성 ─────────────────────────────────────
cat > "$CASE_DIR/system/surfaceFeatureExtractDict" << SFEDICT
FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
netSurface.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj yes;
}
SFEDICT

# ─── 격자 생성 ───────────────────────────────────────────────────────────────
log_step "배경 격자 생성 (blockMesh)"
cd "$CASE_DIR"
blockMesh 2>&1 | tee -a "$LOG_FILE"
log_ok "blockMesh 완료"

log_step "표면 피처 추출 (surfaceFeatureExtract)"
surfaceFeatureExtract 2>&1 | tee -a "$LOG_FILE"
log_ok "surfaceFeatureExtract 완료"

log_step "격자 분할 (decomposePar)"
decomposePar -force 2>&1 | tee -a "$LOG_FILE"
log_ok "decomposePar 완료"

log_step "snappyHexMesh 실행 (병렬, ${N_CORES} 코어)"
mpirun --allow-run-as-root -np "$N_CORES" \
    snappyHexMesh -overwrite -parallel 2>&1 | tee -a "$LOG_FILE"
log_ok "snappyHexMesh 완료"

# ─── CFD 해석 실행 ───────────────────────────────────────────────────────────
log_step "simpleFoam 해석 실행 (병렬, ${N_CORES} 코어)"
log_info "GPU AmgX 가속 자동 감지 중..."

mpirun --allow-run-as-root -np "$N_CORES" \
    simpleFoam -parallel 2>&1 | tee -a "$LOG_FILE"

log_ok "simpleFoam 완료"

# ─── 결과 재조합 ─────────────────────────────────────────────────────────────
log_step "결과 재조합 (reconstructPar)"
reconstructPar -latestTime 2>&1 | tee -a "$LOG_FILE"
log_ok "reconstructPar 완료"

# ─── 결과 추출 (Cd, Cl CSV) ──────────────────────────────────────────────────
log_step "결과 추출 및 CSV 저장"

python3 << PYEOF
import sys, csv, re, os, math
from pathlib import Path
from datetime import datetime

case_dir = Path("$CASE_DIR")
results_dir = Path("$RESULTS_DIR")
output_csv = results_dir / "force_coeffs_unit_cell.csv"

speed = float("$U_SPEED")
angle = float("$AOA")
rho   = 1025.0

# forceCoeffs 데이터 파싱
cd, cl, cm = float('nan'), float('nan'), float('nan')

pp_dir = case_dir / "postProcessing"
for fc_dir in pp_dir.glob("forceCoeffs*"):
    time_dirs = sorted([d for d in fc_dir.iterdir() if d.is_dir()],
                       key=lambda x: float(x.name) if x.name.replace('.','').isdigit() else 0)
    if time_dirs:
        coeff_file = time_dirs[-1] / "forceCoeffs.dat"
        if not coeff_file.exists():
            coeff_file = time_dirs[-1] / "coefficient.dat"
        if coeff_file.exists():
            lines = [l for l in coeff_file.read_text().splitlines()
                     if not l.startswith("#") and l.strip()]
            if lines:
                last = lines[-1].split()
                if len(last) >= 5:
                    cd = float(last[2])
                    cl = float(last[3])
                    cm = float(last[4])

header = ["speed_m_s","angle_deg","Cd","Cl","Cm","Fx_N","Fy_N","Fz_N",
          "rho_kg_m3","case_name","timestamp"]
row = {
    "speed_m_s": speed, "angle_deg": angle,
    "Cd": cd, "Cl": cl, "Cm": cm,
    "Fx_N": "nan", "Fy_N": "nan", "Fz_N": "nan",
    "rho_kg_m3": rho,
    "case_name": case_dir.name,
    "timestamp": datetime.now().isoformat()
}

file_exists = output_csv.exists()
with open(output_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"결과 저장: {output_csv}")
print(f"  Cd = {cd:.4f}, Cl = {cl:.4f}, Cm = {cm:.4f}")
PYEOF

# ─── 완료 요약 ───────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║               해석 완료!                              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
log_ok "케이스 : $CASE_DIR"
log_ok "로그   : $LOG_FILE"
log_ok "CSV    : $RESULTS_DIR/force_coeffs_unit_cell.csv"
log_info "ParaView 시각화: paraFoam -case $CASE_DIR"
