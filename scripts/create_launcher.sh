#!/usr/bin/env bash
# =============================================================================
# create_launcher.sh
# 메인 런처 스크립트 및 바탕화면 아이콘 생성
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$BASE_DIR/.venv"

# ─── 메인 런처 스크립트 생성 ─────────────────────────────────────────────
cat > "$BASE_DIR/launch.sh" << LAUNCHEOF
#!/usr/bin/env bash
# ============================================================
# launch.sh — 양식 가두리 CFD 대시보드 실행
# ============================================================
BASE_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="\$BASE_DIR/.venv"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   🌊 양식 가두리 CFD 대시보드 시작 중...              ║"
echo "╚══════════════════════════════════════════════════════╝"

# Python 가상환경 활성화
if [ -d "\$VENV_DIR" ]; then
    source "\$VENV_DIR/bin/activate"
else
    echo "[경고] 가상환경 미설치. 시스템 Python 사용 중..."
fi

# OpenFOAM 환경 로드
for OF_RC in \
    "/usr/lib/openfoam/openfoam2312/etc/bashrc" \
    "/opt/openfoam10/etc/bashrc" \
    "/opt/openfoam9/etc/bashrc" \
    "\$HOME/OpenFOAM/OpenFOAM-v2312/etc/bashrc"; do
    if [ -f "\$OF_RC" ]; then
        source "\$OF_RC"
        echo "[OK] OpenFOAM 환경 로드: \$OF_RC"
        break
    fi
done

# GPU 환경 설정
if command -v nvidia-smi &>/dev/null; then
    export CUDA_VISIBLE_DEVICES="0,1"
    echo "[OK] GPU 활성화 (RTX 3090 × 2)"
fi

# 포트 사용 여부 확인
PORT=8501
if lsof -Pi :"\$PORT" -sTCP:LISTEN -t &>/dev/null; then
    echo "[INFO] 포트 \$PORT가 이미 사용 중입니다."
    echo "[INFO] http://localhost:\$PORT 를 브라우저에서 열어주세요."
else
    echo "[INFO] Streamlit 대시보드 시작 중 (포트: \$PORT)..."
    cd "\$BASE_DIR"
    streamlit run app.py \\
        --server.port "\$PORT" \\
        --server.headless false \\
        --server.enableCORS false \\
        --browser.gatherUsageStats false &
    
    STREAMLIT_PID=\$!
    echo "[OK] 대시보드 PID: \$STREAMLIT_PID"
    
    # 브라우저 자동 열기 (3초 대기)
    sleep 3
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:\$PORT" &
    elif command -v open &>/dev/null; then
        open "http://localhost:\$PORT" &
    fi
fi

echo ""
echo "🌊 대시보드 URL: http://localhost:\$PORT"
echo "⏹️  종료: Ctrl+C"
echo ""

wait
LAUNCHEOF

chmod +x "$BASE_DIR/launch.sh"
echo "[OK] launch.sh 생성 완료: $BASE_DIR/launch.sh"

# ─── 바탕화면 아이콘 (GNOME .desktop) ────────────────────────────────────
DESKTOP_DIR="$HOME/Desktop"
mkdir -p "$DESKTOP_DIR"

ICON_PATH="$BASE_DIR/assets/cfd_icon.png"

# 아이콘 이미지가 없으면 Python으로 간단한 PNG 생성
if [ ! -f "$ICON_PATH" ]; then
    python3 - << PYEOF
try:
    import PIL.Image as Image, PIL.ImageDraw as Draw
    img = Image.new("RGBA", (128,128), (13,43,94,255))
    draw = Draw.Draw(img)
    draw.ellipse([20,20,108,108], fill=(26,115,232,255), outline=(168,212,253,255), width=3)
    draw.text((32,50), "CFD", fill="white")
    img.save("$ICON_PATH")
except Exception:
    pass
PYEOF
fi

# 메인 대시보드 아이콘
cat > "$DESKTOP_DIR/🌊_CFD_대시보드.desktop" << DESKEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=🌊 가두리 CFD 대시보드
GenericName=CFD Analysis Dashboard
Comment=양식 가두리 OpenFOAM CFD 자동 해석 시스템
Exec=bash -c 'bash "$BASE_DIR/launch.sh"'
Icon=$ICON_PATH
Terminal=false
StartupNotify=true
Categories=Science;Engineering;Education;
Keywords=CFD;OpenFOAM;Aquaculture;FishFarm;
DESKEOF
chmod +x "$DESKTOP_DIR/🌊_CFD_대시보드.desktop"

# ─── /usr/local/bin 심볼릭 링크 (선택사항) ───────────────────────────────
if [ -w "/usr/local/bin" ] || sudo -n true 2>/dev/null; then
    sudo ln -sf "$BASE_DIR/launch.sh" /usr/local/bin/cfd-dashboard 2>/dev/null || true
    echo "[OK] 터미널에서 'cfd-dashboard' 명령으로 실행 가능"
fi

echo "[OK] 런처 및 바탕화면 아이콘 생성 완료"
echo ""
echo "실행 방법:"
echo "  1. 바탕화면의 '🌊 가두리 CFD 대시보드' 더블클릭"
echo "  2. 터미널: bash $BASE_DIR/launch.sh"
echo "  3. 터미널: cfd-dashboard"
