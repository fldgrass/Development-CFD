#!/usr/bin/env bash
# =============================================================================
# run_batch.sh
# 영각 × 유속 조합 배치 해석 자동 실행 스크립트
# Cd/Cl 데이터베이스 CSV 자동 생성
#
# 사용법:
#   ./run_batch.sh [모드] [유속목록] [영각목록] [코어수]
# 예시:
#   ./run_batch.sh unit_cell "0.5 1.0 1.5 2.0" "0 15 30 45 60" 16
#   ./run_batch.sh full_structure "1.0 2.0" "0 30" 16
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_step()  { echo -e "\n${BOLD}${BLUE}═══ $* ═══${NC}"; }

# ─── 인수 처리 ──────────────────────────────────────────────────────────────
MODE="${1:-unit_cell}"                               # unit_cell | full_structure
SPEEDS_STR="${2:-0.5 1.0 1.5 2.0}"                  # 공백 구분 유속 목록
ANGLES_STR="${3:-0 15 30 45 60 75 90}"               # 공백 구분 영각 목록
N_CORES="${4:-0}"
CAGE_D="${5:-10.0}"                                  # 전체 구조 모드용 가두리 직경
CAGE_H="${6:-5.0}"                                   # 전체 구조 모드용 수심

# 배열로 변환
read -ra SPEEDS <<< "$SPEEDS_STR"
read -ra ANGLES <<< "$ANGLES_STR"

# 코어 수 자동 감지
if [ "$N_CORES" -eq 0 ] 2>/dev/null; then
    N_CORES=$(nproc --all)
    N_CORES=$(( N_CORES > 32 ? 32 : N_CORES ))
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$BASE_DIR/results/$MODE"
LOGS_DIR="$BASE_DIR/logs"
BATCH_LOG="$LOGS_DIR/batch_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# 총 케이스 수
TOTAL=$(( ${#SPEEDS[@]} * ${#ANGLES[@]} ))

# ─── 배너 ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     배치 CFD 해석 시스템 — Cd/Cl 데이터베이스 자동 생성   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
log_info "모드        : $MODE"
log_info "유속 목록   : ${SPEEDS[*]} m/s"
log_info "영각 목록   : ${ANGLES[*]} °"
log_info "총 케이스   : $TOTAL개"
log_info "MPI 코어    : $N_CORES개"
log_info "배치 로그   : $BATCH_LOG"

echo "" | tee -a "$BATCH_LOG"
echo "배치 시작: $(date)" | tee -a "$BATCH_LOG"
echo "총 케이스: $TOTAL" | tee -a "$BATCH_LOG"

# ─── 진행률 표시 함수 ────────────────────────────────────────────────────────
show_progress() {
    local current=$1; local total=$2
    local pct=$(( current * 100 / total ))
    local filled=$(( pct / 2 ))
    local bar=""
    for ((i=0; i<50; i++)); do
        if [ $i -lt $filled ]; then bar+="█"; else bar+="░"; fi
    done
    printf "\r${CYAN}진행률: [%s] %d%% (%d/%d)${NC}" "$bar" "$pct" "$current" "$total"
}

# ─── 배치 루프 ──────────────────────────────────────────────────────────────
CASE_COUNT=0
PASS=0
FAIL=0

START_TIME=$SECONDS

for SPEED in "${SPEEDS[@]}"; do
    for ANGLE in "${ANGLES[@]}"; do
        CASE_COUNT=$(( CASE_COUNT + 1 ))
        show_progress $CASE_COUNT $TOTAL
        echo ""

        log_step "케이스 $CASE_COUNT/$TOTAL : U=${SPEED}m/s, α=${ANGLE}°"
        echo "케이스 $CASE_COUNT/$TOTAL: U=$SPEED, α=$ANGLE" >> "$BATCH_LOG"

        # 해석 실행
        set +e
        if [ "$MODE" = "unit_cell" ]; then
            bash "$SCRIPT_DIR/run_unit_cell.sh" \
                "$SPEED" "$ANGLE" "20" "$N_CORES" 2>&1 | tee -a "$BATCH_LOG"
            EXIT_CODE=${PIPESTATUS[0]}
        else
            bash "$SCRIPT_DIR/run_full_structure.sh" \
                "$SPEED" "$ANGLE" "$CAGE_D" "$CAGE_H" "$N_CORES" 2>&1 | tee -a "$BATCH_LOG"
            EXIT_CODE=${PIPESTATUS[0]}
        fi
        set -e

        if [ $EXIT_CODE -eq 0 ]; then
            PASS=$(( PASS + 1 ))
            log_ok "✅ 케이스 완료: U=$SPEED, α=$ANGLE"
            echo "  → 성공" >> "$BATCH_LOG"
        else
            FAIL=$(( FAIL + 1 ))
            log_warn "⚠️ 케이스 실패: U=$SPEED, α=$ANGLE (계속 진행)"
            echo "  → 실패 (exit=$EXIT_CODE)" >> "$BATCH_LOG"
        fi

        # 예상 잔여 시간 계산
        ELAPSED=$(( SECONDS - START_TIME ))
        AVG_TIME=$(( ELAPSED / CASE_COUNT ))
        REMAIN=$(( AVG_TIME * (TOTAL - CASE_COUNT) ))
        log_info "경과: ${ELAPSED}초 | 케이스당 평균: ${AVG_TIME}초 | 잔여: ${REMAIN}초"

    done
done

# ─── 결과 CSV 통합 ───────────────────────────────────────────────────────────
log_step "결과 데이터베이스 통합"

FINAL_CSV="$RESULTS_DIR/force_coeffs_DB_$(date +%Y%m%d_%H%M%S).csv"

python3 << PYEOF
import csv, glob, os
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
final_csv   = Path("$FINAL_CSV")

# 개별 케이스 CSV 통합
header = ["speed_m_s","angle_deg","Cd","Cl","Cm","Fx_N","Fy_N","Fz_N",
          "rho_kg_m3","case_name","timestamp"]
all_rows = []

# 기존 개별 결과 파일들 수집
src_files = list(results_dir.glob("force_coeffs*.csv")) + \
            list(results_dir.glob("results_*.csv"))

for src in sorted(src_files):
    try:
        with open(src) as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
    except Exception as e:
        print(f"읽기 실패: {src}: {e}")

# 속도/영각으로 정렬
try:
    all_rows.sort(key=lambda r: (float(r.get("speed_m_s",0)),
                                  float(r.get("angle_deg",0))))
except Exception:
    pass

# 최종 DB CSV 저장
with open(final_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_rows)

print(f"✅ 데이터베이스 저장: {final_csv}")
print(f"   총 {len(all_rows)}개 레코드")

# 요약 통계 출력
if all_rows:
    cds = [float(r["Cd"]) for r in all_rows if r.get("Cd","nan") not in ("nan","")]
    cls = [float(r["Cl"]) for r in all_rows if r.get("Cl","nan") not in ("nan","")]
    if cds:
        print(f"   Cd 범위: {min(cds):.4f} ~ {max(cds):.4f}")
    if cls:
        print(f"   Cl 범위: {min(cls):.4f} ~ {max(cls):.4f}")
PYEOF

# ─── 배치 완료 요약 ──────────────────────────────────────────────────────────
TOTAL_TIME=$(( SECONDS - START_TIME ))
echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  배치 해석 완료!                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
log_ok "총 케이스    : $TOTAL개"
log_ok "성공         : $PASS개"
[ $FAIL -gt 0 ] && log_warn "실패: $FAIL개" || log_ok "실패: 0개"
log_ok "총 소요 시간 : ${TOTAL_TIME}초 ($(( TOTAL_TIME/60 ))분 $(( TOTAL_TIME%60 ))초)"
log_ok "결과 DB CSV  : $FINAL_CSV"
log_ok "배치 로그    : $BATCH_LOG"
echo ""
log_info "💡 CSV를 질량-스프링 모델(C++)에서 로드하는 방법:"
log_info "   ForceDB db(\"$FINAL_CSV\");"
log_info "   double cd = db.get(speed, angle, \"Cd\");"
