# 🌊 양식 가두리 CFD 해석 시스템 — 완전 설치·운용 매뉴얼

**대상**: 수산공학 전공 교수/연구자  
**환경**: Ubuntu 22.04 LTS / NVIDIA RTX 3090 × 2 / 16+ CPU 코어  
**버전**: v1.0 (2025)

---

## 📋 목차

1. [시스템 개요](#1-시스템-개요)
2. [설치 전 요구사항 확인](#2-설치-전-요구사항-확인)
3. [단계별 설치 가이드](#3-단계별-설치-가이드)
4. [바탕화면 아이콘 설정](#4-바탕화면-아이콘-설정)
5. [단위 셀 모드 사용법](#5-단위-셀-모드-사용법)
6. [전체 구조 모드 사용법](#6-전체-구조-모드-사용법)
7. [배치 해석 사용법](#7-배치-해석-사용법)
8. [GPU 가속 설정](#8-gpu-가속-설정-rtx-3090-×-2)
9. [STL 파일 준비 가이드](#9-stl-파일-준비-가이드-라이노3d)
10. [결과 파일 구조 및 CSV 포맷](#10-결과-파일-구조-및-csv-포맷)
11. [문제 해결](#11-문제-해결-faq)
12. [디렉토리 구조](#12-전체-디렉토리-구조)

---

## 1. 시스템 개요

본 시스템은 **OpenFOAM** 기반의 CFD(전산유체역학) 해석을 **코드 없이 웹 GUI**에서 수행할 수 있도록 설계된 자동화 플랫폼입니다.

```
┌─────────────────────────────────────────────────────────┐
│              양식 가두리 CFD 해석 시스템                  │
├─────────────────┬───────────────────────────────────────┤
│  🔬 단위 셀 모드  │  그물 매듭+그물발 STL → 주기 경계조건  │
│                 │  → Cd/Cl 계수 DB 자동 추출 → CSV        │
├─────────────────┼───────────────────────────────────────┤
│  🏗️ 전체 구조    │  가두리 림+그물 STL → 입구/출구/벽면    │
│     모드        │  → 유동장 해석 → 유속 감쇠 분석 → CSV   │
├─────────────────┴───────────────────────────────────────┤
│  GUI: Streamlit 웹 대시보드 (localhost:8501)             │
│  시각화: PyVista 3D 실시간 렌더링 (10초 자동 갱신)       │
│  엔진: OpenFOAM simpleFoam + MPI 병렬 + GPU AmgX         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 설치 전 요구사항 확인

터미널을 열고 다음 명령을 실행하여 시스템 상태를 확인하세요.

### ✅ 체크리스트

```bash
# (1) Ubuntu 버전 확인
lsb_release -a
# → Ubuntu 20.04 또는 22.04 필요

# (2) GPU 확인
nvidia-smi
# → RTX 3090 × 2 목록이 나와야 함

# (3) CPU 코어 수 확인
nproc --all
# → 권장: 16코어 이상

# (4) 메모리 확인
free -h
# → 권장: 32GB 이상 (최소 16GB)

# (5) 디스크 여유 공간 확인
df -h ~
# → 권장: 100GB 이상 (격자 파일 용량 큼)

# (6) 인터넷 연결 확인
ping -c 3 google.com
```

---

## 3. 단계별 설치 가이드

### 📌 Step 1 — 프로젝트 파일 배치

```bash
# 프로젝트를 홈 디렉토리에 배치합니다.
# (이미 다운로드된 경우 이 단계 건너뜀)

cd ~
# 프로젝트 폴더가 있는 경우:
ls aquaculture_cfd/
```

---

### 📌 Step 2 — OpenFOAM v2312 설치

> ⚠️ **중요**: 이미 OpenFOAM이 설치되어 있으면 Step 3으로 건너뛰세요.

```bash
# ESI OpenFOAM 공식 저장소 추가
wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash

# OpenFOAM v2312 설치 (약 3~5분 소요)
sudo apt-get update
sudo apt-get install openfoam2312

# 환경 변수 영구 등록
echo "source /usr/lib/openfoam/openfoam2312/etc/bashrc" >> ~/.bashrc
source ~/.bashrc

# 설치 확인
blockMesh -help
# → "Usage: blockMesh [OPTIONS]" 메시지가 나와야 함
```

> 💡 **대안**: Foundation 버전(openfoam.org)도 사용 가능합니다.
> ```bash
> sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
> sudo add-apt-repository http://dl.openfoam.org/ubuntu
> sudo apt-get install openfoam10
> echo "source /opt/openfoam10/etc/bashrc" >> ~/.bashrc
> ```

---

### 📌 Step 3 — MPI (병렬 연산) 설치

```bash
# OpenMPI 설치 (이미 설치된 경우 건너뜀)
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# 설치 확인
mpirun --version
# → "Open MPI) 4.x.x" 또는 유사한 출력

# MPI 테스트 (4코어 테스트)
mpirun -np 4 echo "MPI 코어 테스트"
```

---

### 📌 Step 4 — Python 환경 설치

```bash
# 프로젝트 디렉토리로 이동
cd ~/aquaculture_cfd

# Python 가상환경 생성 (시스템 Python과 분리)
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 필수 패키지 설치 (약 5~10분 소요)
pip install -r requirements.txt

# 설치 확인
python3 -c "import streamlit, pyvista, numpy; print('✅ 모든 패키지 정상')"
```

> 💡 **주의**: PyVista 설치 시 VTK가 함께 설치됩니다. 용량이 크므로(~300MB) 시간이 걸릴 수 있습니다.

---

### 📌 Step 5 — 자동 설치 스크립트 실행 (통합)

위 모든 단계를 한 번에 실행하려면:

```bash
cd ~/aquaculture_cfd
chmod +x scripts/install.sh
bash scripts/install.sh
```

설치 로그는 `logs/install_날짜.log`에 자동 저장됩니다.

---

### 📌 Step 6 — 설치 최종 확인

```bash
# 모든 설치 검증
cd ~/aquaculture_cfd
source .venv/bin/activate

python3 - << 'EOF'
import sys
checks = {}

# OpenFOAM
import subprocess, shutil
checks['OpenFOAM'] = bool(shutil.which('blockMesh'))
checks['MPI']      = bool(shutil.which('mpirun'))

# Python 패키지
try:
    import streamlit; checks['Streamlit'] = True
except: checks['Streamlit'] = False

try:
    import pyvista; checks['PyVista'] = True
except: checks['PyVista'] = False

try:
    import numpy; checks['NumPy'] = True
except: checks['NumPy'] = False

try:
    import pandas; checks['Pandas'] = True
except: checks['Pandas'] = False

# GPU
try:
    result = subprocess.run(['nvidia-smi','--query-gpu=name',
                             '--format=csv,noheader'],
                            capture_output=True, text=True)
    gpus = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    checks[f'GPU ({len(gpus)}개)'] = len(gpus) > 0
except: checks['GPU'] = False

print("=" * 45)
print("   설치 상태 점검 결과")
print("=" * 45)
for name, ok in checks.items():
    status = "✅ 설치됨" if ok else "❌ 미설치"
    print(f"  {status}  {name}")
print("=" * 45)
EOF
```

---

## 4. 바탕화면 아이콘 설정

### 자동 생성 (권장)

```bash
cd ~/aquaculture_cfd
bash scripts/create_launcher.sh
```

4개의 아이콘이 바탕화면에 자동 생성됩니다:

| 아이콘 | 기능 |
|--------|------|
| 🌊 가두리 CFD 대시보드 | Streamlit 웹 대시보드 실행 |
| 🔬 단위 셀 해석 | 단위 셀 모드 직접 실행 |
| 🏗️ 전체 구조 해석 | 전체 구조 모드 직접 실행 |
| 🔄 배치 해석 | 자동 배치 해석 실행 |

### GNOME 바탕화면 아이콘 활성화 (Ubuntu 22.04)

```bash
# GNOME에서 .desktop 파일을 신뢰 가능하게 설정
gio set ~/Desktop/🌊_CFD_대시보드.desktop \
    metadata::trusted true

# 권한 부여
chmod +x ~/Desktop/*.desktop
```

> 💡 아이콘이 보이지 않을 경우:  
> 바탕화면 우클릭 → "Display Settings" → "Show Desktop Icons" 활성화

### 터미널에서 직접 실행

```bash
# 방법 1: launch.sh 실행
bash ~/aquaculture_cfd/launch.sh

# 방법 2: 직접 Streamlit 실행
cd ~/aquaculture_cfd
source .venv/bin/activate
streamlit run app.py --server.port 8501

# 방법 3: 심볼릭 링크 (install.sh 실행 시 자동 설정)
cfd-dashboard
```

브라우저에서 **http://localhost:8501** 접속

---

## 5. 단위 셀 모드 사용법

### 목적
- 그물 1개 단위 셀(매듭 + 그물발)의 유체력 계수 추출
- 주기 경계조건(Cyclic) 자동 적용
- Cd(항력계수), Cl(양력계수) 데이터베이스 구축

### ① STL 파일 준비 (Rhino3D)

```
Rhino3D에서:
1. 단위 셀(매듭 1개 + 연결 그물발) 3D 모델 완성
2. 메뉴: File → Export Selected
3. 파일 형식: Sterolithography (.stl)
4. 설정: Binary STL, 공차 0.001mm
5. 법선 방향 확인: Analyze → Direction → 외부 방향(빨간색)이어야 함
```

> ⚠️ **중요**: STL 파일의 법선 벡터가 바깥쪽을 향해야 합니다.  
> 반전된 경우: `Mesh → Repair → Unify Normals` 실행

### ② 웹 대시보드에서 실행

1. 브라우저에서 **http://localhost:8501** 접속
2. 좌측 사이드바 → **"🔬 단위 셀 모드"** 선택
3. **"📂 입력 설정"** 탭 클릭
4. **그물 단위 셀 STL** 파일 업로드
5. 해석 파라미터 입력:

| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| 유속 U | 0.5 ~ 2.0 m/s | 조류 유속 |
| 영각 AoA | 0 ~ 90° | 유입각 |
| 단위 셀 크기 a | 10 ~ 50 mm | 메쉬 사이즈 |
| MPI 코어 수 | 16 | CPU 코어 수 |
| 최대 반복 횟수 | 2000 | 수렴 기준 |

6. **"▶️ 해석 시작"** 클릭

### ③ 주기 경계조건 자동 설정 원리

```
단위 셀 도메인:
  ┌──────────────┐
  │   yMax(cyclic)│
  │  ┌──────┐    │
xMin│  │ 매듭 │    │xMax
(cy)│  │+그물발│    │(cy)
  │  └──────┘    │
  │   yMin(cyclic)│
  └──────────────┘
  zMin(inlet) → zMax(outlet)

자동 설정 로직 (cfd_manager.py):
  xMin ↔ xMax: cyclic (x방향 반복)
  yMin ↔ yMax: cyclic (y방향 반복)
  zMin: fixedValue (유속 입구)
  zMax: zeroGradient (압력 출구)
  netSurface: noSlip (그물 표면)
```

### ④ 셸 스크립트 직접 실행

```bash
# 형식: ./run_unit_cell.sh [유속] [영각] [셀크기mm] [코어수]
cd ~/aquaculture_cfd
bash scripts/run_unit_cell.sh 1.0 0 20 16
bash scripts/run_unit_cell.sh 1.5 30 20 16
```

---

## 6. 전체 구조 모드 사용법

### 목적
- 원통형 양식 가두리 전체의 유동장 해석
- 입구(Inlet)/출구(Outlet)/벽면(Wall) 경계조건 자동 설정
- 가두리 내·외부 유속 감쇠율 계산

### ① STL 파일 준비

```
필요한 STL 파일 2개:
  1. cageSurface.stl  → 금속 림(Rim), 부력체, 프레임
  2. netSurface.stl   → 그물망 전체

Rhino3D에서:
  - 가두리 림과 그물을 별도 레이어로 분리
  - 각각 STL로 내보내기
  - 단위: 미터(m) 권장
  - 원점(0,0,0): 가두리 중심
```

### ② 웹 대시보드에서 실행

1. 좌측 사이드바 → **"🏗️ 전체 구조 모드"** 선택
2. **가두리 림 STL** 및 **그물 STL** 각각 업로드
3. 가두리 치수 입력:

| 파라미터 | 예시값 | 설명 |
|----------|--------|------|
| 가두리 직경 D | 10 m | 원통형 가두리 직경 |
| 가두리 수심 H | 5 m | 그물망 깊이 |
| 유속 | 1.0 m/s | 조류 유속 |

4. **"▶️ 해석 시작"** 클릭

### ③ 도메인 자동 구성

```
유동 도메인 (D=10m 기준):
  x축: -30m ~ +70m  (상류 3D + 하류 7D)
  y축: -30m ~ +30m  (횡방향 3D)
  z축: -5m ~ 0m     (수심 H)

경계조건 자동 할당:
  inlet  (x_min): fixedValue → 입력 유속
  outlet (x_max): zeroGradient + inletOutlet
  top/bottom/sides: slip (자유 슬립)
  cageSurface: noSlip + kqRWallFunction
  netSurface:  noSlip + kqRWallFunction
```

### ④ 셸 스크립트 직접 실행

```bash
# 형식: ./run_full_structure.sh [유속] [영각] [직경m] [수심m] [코어수]
bash scripts/run_full_structure.sh 1.0 0 10 5 16
```

---

## 7. 배치 해석 사용법

### 목적
Cd/Cl 데이터베이스를 자동 구축합니다.  
여러 유속×영각 조합을 **자동 순환** 실행하여 결과를 **1개의 CSV**로 통합합니다.

### 웹 대시보드에서 실행

1. **"🔄 배치 해석"** 탭 클릭
2. 유속 범위 설정 (예: 0.5 ~ 2.0 m/s, 4단계)
3. 영각 범위 설정 (예: 0 ~ 90°, 7단계)
4. 총 케이스 수 확인 (4 × 7 = 28개)
5. **"🚀 배치 해석 시작"** 클릭

### 셸 스크립트 직접 실행

```bash
# 형식:
# ./run_batch.sh [모드] [유속목록] [영각목록] [코어수]

# 단위 셀 배치 (유속 4개 × 영각 7개 = 28케이스)
bash scripts/run_batch.sh \
    unit_cell \
    "0.5 1.0 1.5 2.0" \
    "0 15 30 45 60 75 90" \
    16

# 전체 구조 배치
bash scripts/run_batch.sh \
    full_structure \
    "0.5 1.0 2.0" \
    "0 30 60" \
    16
```

### 예상 소요 시간 (참고)

| 격자 수 | 코어 16개 | GPU 가속 추가 |
|---------|-----------|--------------|
| 100만 셀 | ~10분/케이스 | ~5분/케이스 |
| 500만 셀 | ~40분/케이스 | ~20분/케이스 |
| 1000만 셀 | ~90분/케이스 | ~45분/케이스 |

---

## 8. GPU 가속 설정 (RTX 3090 × 2)

### 8-1. CUDA 확인

```bash
nvidia-smi
# RTX 3090 × 2가 표시되어야 함
# CUDA Version: 12.x 필요

nvcc --version
# CUDA 11.8 이상 필요 (RTX 3090: Compute Capability 8.6)
```

### 8-2. AmgX 빌드 및 설치

AmgX는 NVIDIA의 GPU 기반 행렬 방정식 솔버로, OpenFOAM의 GAMG 솔버를 대체하여 **선형 방정식 풀이 속도를 5~10배** 향상시킵니다.

```bash
# 1. AmgX 소스 다운로드
cd ~
git clone https://github.com/NVIDIA/AMGX.git

# 2. 빌드 디렉토리 생성
cd AMGX && mkdir build && cd build

# 3. CMake 구성 (RTX 3090: sm_86)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH="86"              \  # RTX 3090 Compute Capability
    -DCMAKE_INSTALL_PREFIX=~/amgx-install

# 4. 빌드 (약 20~30분)
make -j$(nproc) && make install

# 5. 환경 변수 등록
echo "export AMGX_DIR=~/amgx-install" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$AMGX_DIR/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### 8-3. OpenFOAM-AmgX 플러그인 설치

```bash
# AmgX4Foam 플러그인
cd ~
git clone https://github.com/NVIDIA/AMGX4Foam.git
cd AMGX4Foam
source /usr/lib/openfoam/openfoam2312/etc/bashrc
wmake

# fvSolution에서 AmgX 솔버 활성화:
# solvers { p { solver amgxSolver; ... } }
```

### 8-4. MPI + GPU 동시 활용 전략

```
RTX 3090 × 2 최적 구성:
  CPU: 16코어 → MPI 16 프로세스
  GPU: RTX3090 #0 → MPI 프로세스 0~7 담당
       RTX3090 #1 → MPI 프로세스 8~15 담당
  
  환경변수: CUDA_VISIBLE_DEVICES=0,1
  효과: 선형 솔버(압력 방정식)만 GPU 오프로드
        나머지(대류항 계산)는 CPU MPI 병렬
```

### 8-5. GPU 없이 CPU만 사용하는 경우

GPU가 없거나 AmgX 미설치 시, **GAMG 솔버(CPU)**가 자동으로 사용됩니다.  
fvSolution의 현재 설정:
```cpp
p { solver GAMG; smoother GaussSeidel; ... }
```
이 설정은 CPU 16코어에서도 효과적으로 동작합니다.

---

## 9. STL 파일 준비 가이드 (Rhino3D)

### 단위 셀 STL 준비

```
[Rhino3D 작업 순서]

1. 새 파일 열기 (단위: mm)

2. 그물발(Twine) 모델링:
   - 명령어: _Cylinder
   - 직경: 그물발 직경 d (예: 1.0mm)
   - 중심점: (0, 0, -10) → (0, 0, 10)

3. 매듭(Knot) 모델링:
   - 명령어: _Sphere
   - 중심: (0, 0, 0), 반경: d × 1.5

4. 메시 변환:
   - 선택 → Mesh → From NURBS Object
   - Maximum edge length: 0.2mm

5. 법선 방향 확인:
   - Analyze → Direction
   - 빨간색 화살표가 외부 방향이어야 함
   - 반전: Mesh → Repair → Unify Normals

6. STL 내보내기:
   - File → Export Selected
   - Format: STL (Binary)
   - Tolerance: 0.001 mm
   - 파일명: net_unitcell.stl
```

### 전체 가두리 STL 준비

```
[가두리 림 STL]
   - 원형 프레임(토러스): Torus 명령
   - 단위: m
   - 중심: (0, 0, 0)
   - 반경: D/2 (예: 5.0m)
   - 내보내기: cageSurface.stl

[그물 STL]
   - 원통형 그물망 (shell): Cylinder 명령
   - 반경: D/2, 높이: H
   - 두께 무시 (쉘로 표현 가능)
   - 내보내기: netSurface.stl
```

> 💡 **팁**: 실제 그물의 공극률(solidity)을 반영하려면  
> `porousMedia` 또는 `Darcy-Forchheimer` 모델 적용을 검토하세요.

---

## 10. 결과 파일 구조 및 CSV 포맷

### 결과 디렉토리 구조

```
results/
├── unit_cell/
│   ├── unit_cell_U1.00_A0.0_20250101_120000/   ← 케이스별 OpenFOAM 결과
│   │   ├── 0/            ← 초기 조건
│   │   ├── 1000/         ← 중간 결과
│   │   ├── 2000/         ← 최종 결과
│   │   ├── constant/
│   │   ├── system/
│   │   └── postProcessing/
│   │       └── forceCoeffs/
│   │           └── 0/
│   │               └── forceCoeffs.dat    ← Cd, Cl 원본 데이터
│   ├── force_coeffs_unit_cell.csv         ← 개별 케이스 결과
│   └── force_coeffs_DB_20250101.csv       ← 배치 통합 데이터베이스
│
└── full_structure/
    ├── full_struct_U1.00_A0.0_.../
    │   └── postProcessing/
    │       ├── forceCoeffs_cage/
    │       └── velocitySampling/          ← 유속 감쇠 데이터
    └── results_full_structure.csv
```

### CSV 포맷 (질량-스프링 모델 호환)

```csv
speed_m_s,angle_deg,Cd,Cl,Cm,Fx_N,Fy_N,Fz_N,rho_kg_m3,case_name,timestamp
0.5,0.0,0.2341,0.0123,0.0012,1.23,0.06,0.01,1025.0,unit_cell_U0.50_A0.0,2025-01-01T12:00:00
0.5,15.0,0.2412,0.0891,0.0034,1.27,0.47,0.18,1025.0,unit_cell_U0.50_A15.0,2025-01-01T12:30:00
...
```

| 컬럼 | 단위 | 설명 |
|------|------|------|
| speed_m_s | m/s | 유속 크기 |
| angle_deg | ° | 영각(Attack of Angle) |
| Cd | - | 항력 계수 |
| Cl | - | 양력 계수 |
| Cm | - | 모멘트 계수 |
| Fx_N | N | x방향 힘 (항력) |
| Fy_N | N | y방향 힘 |
| Fz_N | N | z방향 힘 (양력) |
| rho_kg_m3 | kg/m³ | 해수 밀도 |
| case_name | - | 케이스 식별자 |
| timestamp | - | 해석 완료 시각 |

### C++ 질량-스프링 모델에서 CSV 로드 예시

```cpp
#include <fstream>
#include <sstream>
#include <map>
#include <tuple>

class ForceCoeffDB {
public:
    // CSV 로드
    void load(const std::string& csv_path) {
        std::ifstream f(csv_path);
        std::string line;
        std::getline(f, line); // 헤더 건너뜀
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::string> cols;
            while (std::getline(ss, tok, ',')) cols.push_back(tok);
            if (cols.size() < 5) continue;
            double speed = std::stod(cols[0]);
            double angle = std::stod(cols[1]);
            double cd    = std::stod(cols[2]);
            double cl    = std::stod(cols[3]);
            db_[{speed, angle}] = {cd, cl};
        }
    }
    
    // Cd/Cl 조회 (선형 보간)
    double getCd(double speed, double angle) const {
        auto it = db_.find({speed, angle});
        if (it != db_.end()) return it->second.first;
        return interpolate(speed, angle, 0); // 선형 보간
    }
    double getCl(double speed, double angle) const {
        auto it = db_.find({speed, angle});
        if (it != db_.end()) return it->second.second;
        return interpolate(speed, angle, 1);
    }

private:
    std::map<std::pair<double,double>, std::pair<double,double>> db_;
    double interpolate(double speed, double angle, int idx) const { /*...*/ return 0; }
};

// 사용 예시
ForceCoeffDB db;
db.load("/path/to/force_coeffs_DB.csv");
double Cd = db.getCd(1.5, 30.0);
double Cl = db.getCl(1.5, 30.0);
```

---

## 11. 문제 해결 FAQ

### ❌ "blockMesh: command not found"

```bash
# OpenFOAM 환경이 로드되지 않은 경우
source /usr/lib/openfoam/openfoam2312/etc/bashrc
# 또는
source /opt/openfoam10/etc/bashrc

# 영구 해결: ~/.bashrc에 위 줄 추가
```

### ❌ "mpirun: command not found"

```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

### ❌ snappyHexMesh 실패 — "zero cells"

```bash
# STL 법선 방향 문제일 가능성이 높음
# 해결: 도메인 내부의 locationInMesh 좌표 확인
# system/snappyHexMeshDict의 locationInMesh가
# 도메인 내부 (STL 외부)에 있어야 함

# STL 유효성 검사
surfaceCheck constant/triSurface/netSurface.stl
```

### ❌ PyVista 렌더링 오류

```bash
# 헤드리스 서버 환경인 경우 (디스플레이 없음)
export DISPLAY=:0
Xvfb :0 -screen 0 1024x768x24 &

# 또는 오프스크린 렌더링 강제
pip install pyvista[all] --force-reinstall
```

### ❌ "simpleFoam: diverged"

```
수렴 불안정 해결 방법:
1. fvSolution 완화 계수 줄이기:
   U: 0.7 → 0.5
   p: 0.3 → 0.2

2. 격자 품질 확인:
   checkMesh -latestTime
   → maxNonOrtho < 70, maxSkewness < 4 이어야 함

3. 유속 단계적으로 증가:
   시작: 0.1 m/s → 점진적으로 목표 유속까지 증가
   (multipleRun 또는 초기 조건 수정)
```

### ❌ "Permission denied" (MPI)

```bash
# Docker/Root 환경에서 MPI 실행 시
mpirun --allow-run-as-root -np 16 simpleFoam -parallel
```

### ❌ "Out of memory" (격자 생성 중)

```bash
# snappyHexMesh 메모리 제한 완화
# system/snappyHexMeshDict에서:
maxGlobalCells  10000000;  # 줄이기: 5000000

# 스왑 메모리 확보
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 12. 전체 디렉토리 구조

```
aquaculture_cfd/
│
├── app.py                          ← Streamlit 메인 대시보드
├── launch.sh                       ← 원클릭 실행 스크립트
├── requirements.txt                ← Python 의존성
├── MANUAL.md                       ← 이 문서
│
├── modules/
│   ├── __init__.py
│   ├── cfd_manager.py              ← CFD 해석 관리 백엔드
│   └── visualizer.py               ← PyVista 3D 시각화
│
├── openfoam/
│   ├── unit_cell/                  ← 단위 셀 모드 템플릿
│   │   ├── 0/
│   │   │   ├── U                   ← 속도 경계조건 (cyclic)
│   │   │   ├── p                   ← 압력 경계조건
│   │   │   ├── k                   ← 난류 운동에너지
│   │   │   ├── omega               ← 비소산율
│   │   │   └── nut                 ← 난류 점성계수
│   │   ├── constant/
│   │   │   ├── transportProperties ← 해수 물성치
│   │   │   ├── turbulenceProperties← k-omega SST 설정
│   │   │   └── triSurface/         ← STL 파일 저장 위치
│   │   └── system/
│   │       ├── blockMeshDict       ← 배경 격자 (단위 셀 크기 자동)
│   │       ├── snappyHexMeshDict   ← 격자 정밀화 설정
│   │       ├── controlDict         ← 해석 제어 + forceCoeffs
│   │       ├── fvSchemes           ← 수치 기법
│   │       ├── fvSolution          ← 선형 솔버 설정 (GAMG/AmgX)
│   │       └── decomposeParDict    ← MPI 도메인 분할
│   │
│   └── full_structure/             ← 전체 구조 모드 템플릿
│       ├── 0/  (U, p, k, omega, nut)
│       ├── constant/
│       └── system/
│           ├── blockMeshDict       ← 도메인 크기 자동 계산
│           ├── snappyHexMeshDict   ← 가두리 주변 정밀화
│           ├── controlDict         ← forceCoeffs + velocitySampling
│           ├── fvSchemes, fvSolution, decomposeParDict
│
├── scripts/
│   ├── install.sh                  ← 통합 설치 스크립트
│   ├── create_launcher.sh          ← 바탕화면 아이콘 생성
│   ├── run_unit_cell.sh            ← 단위 셀 해석 스크립트
│   ├── run_full_structure.sh       ← 전체 구조 해석 스크립트
│   ├── run_batch.sh                ← 배치 해석 스크립트
│   └── patch_blockMesh_unit.py     ← blockMeshDict 패치 헬퍼
│
├── stl_uploads/                    ← 업로드된 STL 임시 저장
├── results/
│   ├── unit_cell/                  ← 단위 셀 결과 및 CSV
│   └── full_structure/             ← 전체 구조 결과 및 CSV
├── logs/                           ← 해석 로그 파일
└── assets/                         ← UI 이미지 리소스
```

---

## 📞 지원

- **OpenFOAM 공식 문서**: https://www.openfoam.com/documentation
- **OpenFOAM 포럼**: https://www.cfd-online.com/Forums/openfoam/
- **PyVista 문서**: https://docs.pyvista.org
- **Streamlit 문서**: https://docs.streamlit.io

---

*본 시스템은 수산공학 CFD 연구 목적으로 설계되었습니다.*  
*OpenFOAM® 는 OpenCFD Ltd.의 등록 상표입니다.*
