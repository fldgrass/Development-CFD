#!/usr/bin/env bash
# =============================================================================
# run_full_structure.sh
# 전체 구조(Full Structure) 해석 자동 실행 스크립트
# 원통형 가두리 유동장 해석 + 유속 감쇠 분석
#
# 사용법:
#   ./run_full_structure.sh [유속] [영각] [직경m] [수심m] [코어수]
# 예시:
#   ./run_full_structure.sh 1.0 0 10 5 16
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${BLUE}▶ $*${NC}"; }

# ─── 인수 처리 ──────────────────────────────────────────────────────────────
U_SPEED="${1:-1.0}"
AOA="${2:-0}"
CAGE_D="${3:-10.0}"
CAGE_H="${4:-5.0}"
N_CORES="${5:-0}"

if [ "$N_CORES" -eq 0 ] 2>/dev/null; then
    N_CORES=$(nproc --all)
    N_CORES=$(( N_CORES > 32 ? 32 : N_CORES ))
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATE_DIR="$BASE_DIR/openfoam/full_structure"
RESULTS_DIR="$BASE_DIR/results/full_structure"
LOGS_DIR="$BASE_DIR/logs"
STL_DIR="$BASE_DIR/stl_uploads"

CASE_NAME="full_struct_U${U_SPEED}_A${AOA}_$(date +%Y%m%d_%H%M%S)"
CASE_DIR="$RESULTS_DIR/$CASE_NAME"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/${CASE_NAME}.log"

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   양식 가두리 전체 구조 CFD 해석 시스템                ║"
echo "║   OpenFOAM + MPI + GPU(AmgX) 자동화 스크립트          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
log_info "유속     : $U_SPEED m/s | 영각: $AOA° | D=$CAGE_D m | H=$CAGE_H m"
log_info "MPI 코어 : $N_CORES개"

# ─── OpenFOAM 환경 로드 ─────────────────────────────────────────────────────
log_step "OpenFOAM 환경 초기화"
for candidate in \
    "/opt/openfoam10/etc/bashrc" "/opt/openfoam9/etc/bashrc" \
    "/usr/lib/openfoam/openfoam2312/etc/bashrc" \
    "$HOME/OpenFOAM/OpenFOAM-v2312/etc/bashrc"; do
    if [ -f "$candidate" ]; then
        # shellcheck disable=SC1090
        source "$candidate" && log_ok "OpenFOAM: $candidate" && break
    fi
done
command -v blockMesh &>/dev/null || { log_error "OpenFOAM 미설치"; exit 1; }

# ─── GPU 환경 ────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    export CUDA_VISIBLE_DEVICES="0,1"
    log_ok "GPU RTX 3090 × 2 활성화 (CUDA_VISIBLE_DEVICES=0,1)"
fi

# ─── 케이스 구성 ─────────────────────────────────────────────────────────────
log_step "케이스 디렉토리 구성"
cp -r "$TEMPLATE_DIR" "$CASE_DIR"

TRISURF_DIR="$CASE_DIR/constant/triSurface"
mkdir -p "$TRISURF_DIR"

# cage STL 복사
CAGE_STL=$(ls -t "$STL_DIR"/cage*.stl 2>/dev/null | head -1 || echo "")
NET_STL=$(ls  -t "$STL_DIR"/net*.stl  2>/dev/null | head -1 || echo "")

if [ -n "$CAGE_STL" ]; then
    cp "$CAGE_STL" "$TRISURF_DIR/cageSurface.stl"
    log_ok "가두리 STL: $(basename "$CAGE_STL")"
else
    log_warn "cage STL 미발견 → 더미 STL 사용"
    printf "solid cageSurface\nendsolid cageSurface\n" > "$TRISURF_DIR/cageSurface.stl"
fi

if [ -n "$NET_STL" ]; then
    cp "$NET_STL" "$TRISURF_DIR/netSurface.stl"
    log_ok "그물 STL: $(basename "$NET_STL")"
else
    log_warn "net STL 미발견 → 더미 STL 사용"
    printf "solid netSurface\nendsolid netSurface\n" > "$TRISURF_DIR/netSurface.stl"
fi

# ─── Python으로 전체 케이스 패치 ─────────────────────────────────────────────
log_step "경계조건 및 도메인 자동 설정 (Python)"
python3 << PYEOF
import math, re
from pathlib import Path

case_dir = Path("$CASE_DIR")
speed = float("$U_SPEED")
aoa   = math.radians(float("$AOA"))
D     = float("$CAGE_D")
H     = float("$CAGE_H")
n     = int("$N_CORES")

# 속도 벡터
Ux = speed * math.cos(aoa)
Uz = speed * math.sin(aoa)
Uvec = f"({Ux:.6f} 0.000000 {Uz:.6f})"

# 난류 파라미터
k     = 1.5 * (0.05 * speed)**2
omega = math.sqrt(k) / (0.09**0.25 * D * 0.1)
print(f"속도벡터: {Uvec}")
print(f"k={k:.4e}, omega={omega:.4f}")

# 도메인 크기 계산
x_min = -3*D;  x_max = 7*D
y_min = -3*D;  y_max = 3*D
z_min = -H;    z_max = 0.0
nx = max(50, int(10*D)); ny = max(40, int(6*D)); nz = max(10, int(4*H))

def replace(path, reps):
    t = path.read_text()
    for o, n in reps.items():
        t = t.replace(o, n)
    path.write_text(t)

# U 경계조건
replace(case_dir/"0"/"U", {
    "uniform (1.0 0 0);": f"uniform {Uvec};",
    "uniform (0 0 0);":   f"uniform {Uvec};",
})

# 난류 초기조건
for fname, val in [("k", f"{k:.6e}"), ("omega", f"{omega:.4f}")]:
    fpath = case_dir/"0"/fname
    text = fpath.read_text()
    text = re.sub(r"uniform\s+[\d.e+-]+;", f"uniform {val};", text)
    fpath.write_text(text)

# blockMeshDict 도메인 수정
replace(case_dir/"system"/"blockMeshDict", {
    "(-30  -30  -5)": f"({x_min:.1f}  {y_min:.1f}  {z_min:.1f})",
    "( 70  -30  -5)": f"({x_max:.1f}  {y_min:.1f}  {z_min:.1f})",
    "( 70   30  -5)": f"({x_max:.1f}  {y_max:.1f}  {z_min:.1f})",
    "(-30   30  -5)": f"({x_min:.1f}  {y_max:.1f}  {z_min:.1f})",
    "(-30  -30   0)": f"({x_min:.1f}  {y_min:.1f}  {z_max:.1f})",
    "( 70  -30   0)": f"({x_max:.1f}  {y_min:.1f}  {z_max:.1f})",
    "( 70   30   0)": f"({x_max:.1f}  {y_max:.1f}  {z_max:.1f})",
    "(-30   30   0)": f"({x_min:.1f}  {y_max:.1f}  {z_max:.1f})",
    "(100 60 20)":    f"({nx} {ny} {nz})",
})

# snappyHexMesh 정밀화 박스
r = D/2 * 1.2
replace(case_dir/"system"/"snappyHexMeshDict", {
    "min     (-6 -6 -6);": f"min     ({-r:.2f} {-r:.2f} {-(H+1):.2f});",
    "max     ( 6  6  1);": f"max     ({r:.2f}  {r:.2f}  1.0);",
    "locationInMesh (0 0 -2.5);": f"locationInMesh (0 0 {-H/2:.2f});",
})

# controlDict 기준값
aref = D * H
replace(case_dir/"system"/"controlDict", {
    "magUInf         1.0;": f"magUInf         {speed:.4f};",
    "lRef            10.0;": f"lRef            {D:.4f};",
    "Aref            50.0;": f"Aref            {aref:.4f};",
})

# decomposeParDict
replace(case_dir/"system"/"decomposeParDict", {
    "numberOfSubdomains  16;": f"numberOfSubdomains  {n};",
})

print("케이스 패치 완료")
PYEOF

# surfaceFeatureExtractDict
cat > "$CASE_DIR/system/surfaceFeatureExtractDict" << 'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
cageSurface.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj yes;
}
netSurface.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj yes;
}
EOF

# ─── 격자 생성 단계 ──────────────────────────────────────────────────────────
cd "$CASE_DIR"

log_step "blockMesh"; blockMesh 2>&1 | tee -a "$LOG_FILE"; log_ok "완료"

log_step "surfaceFeatureExtract"
surfaceFeatureExtract 2>&1 | tee -a "$LOG_FILE"; log_ok "완료"

log_step "decomposePar"
decomposePar -force 2>&1 | tee -a "$LOG_FILE"; log_ok "완료"

log_step "snappyHexMesh (병렬 $N_CORES 코어)"
mpirun --allow-run-as-root -np "$N_CORES" \
    snappyHexMesh -overwrite -parallel 2>&1 | tee -a "$LOG_FILE"
log_ok "snappyHexMesh 완료"

# ─── CFD 해석 ────────────────────────────────────────────────────────────────
log_step "simpleFoam 병렬 해석 ($N_CORES 코어, GPU 가속)"
mpirun --allow-run-as-root -np "$N_CORES" \
    simpleFoam -parallel 2>&1 | tee -a "$LOG_FILE"
log_ok "simpleFoam 완료"

log_step "reconstructPar"
reconstructPar -latestTime 2>&1 | tee -a "$LOG_FILE"
log_ok "reconstructPar 완료"

# ─── 결과 추출 ───────────────────────────────────────────────────────────────
log_step "유속 감쇠 분석 및 결과 CSV 저장"
python3 << PYEOF
import csv, re
from pathlib import Path
from datetime import datetime

case_dir    = Path("$CASE_DIR")
results_dir = Path("$RESULTS_DIR")
output_csv  = results_dir / "results_full_structure.csv"
speed = float("$U_SPEED"); angle = float("$AOA"); rho = 1025.0

# forceCoeffs 파싱
cd, cl, cm = float('nan'), float('nan'), float('nan')
for fc_dir in (case_dir/"postProcessing").glob("forceCoeffs*"):
    tdirs = sorted([d for d in fc_dir.iterdir() if d.is_dir()],
                   key=lambda x: float(x.name) if x.name.replace('.','').isdigit() else 0)
    if tdirs:
        cf = tdirs[-1]/"forceCoeffs.dat"
        if not cf.exists(): cf = tdirs[-1]/"coefficient.dat"
        if cf.exists():
            lines = [l for l in cf.read_text().splitlines()
                     if not l.startswith("#") and l.strip()]
            if lines:
                parts = lines[-1].split()
                if len(parts) >= 5:
                    cd, cl, cm = float(parts[2]), float(parts[3]), float(parts[4])

header = ["speed_m_s","angle_deg","Cd","Cl","Cm","Fx_N","Fy_N","Fz_N",
          "rho_kg_m3","case_name","timestamp"]
row = {"speed_m_s":speed,"angle_deg":angle,"Cd":cd,"Cl":cl,"Cm":cm,
       "Fx_N":"nan","Fy_N":"nan","Fz_N":"nan","rho_kg_m3":rho,
       "case_name":case_dir.name,"timestamp":datetime.now().isoformat()}

fe = output_csv.exists()
with open(output_csv,"a",newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    if not fe: w.writeheader()
    w.writerow(row)

print(f"✅ 저장: {output_csv}")
print(f"   Cd={cd:.4f}  Cl={cl:.4f}  Cm={cm:.4f}")
PYEOF

echo -e "\n${BOLD}${GREEN}✅ 전체 구조 해석 완료!${NC}"
log_ok "케이스: $CASE_DIR"
log_ok "로그  : $LOG_FILE"
log_ok "CSV   : $RESULTS_DIR/results_full_structure.csv"
