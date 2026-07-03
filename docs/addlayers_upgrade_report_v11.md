# addLayers(경계층 프리즘 격자) 업그레이드 보고서 (v11)

날짜: 2026-07-03 · 목적: 벤치마크에서 정량화된 그물실 항력 ~2배 저평가(벽 격자
0.31mm ≫ 경계층 δ≈0.06mm)의 근본 해결. 과거 getFaceOrder FATAL 로 비활성화됐던
addLayers 를 재도입.

## 1. 벤치마크(고립 원기둥 d=3mm, Re=2,500) 단계 검증

| 격자 | Cd | 실험(≈1.0) 대비 | 비고 |
|---|---|---|---|
| 종전(벽 0.31mm, 무레이어) | 0.502 | −50% | 방출 없음 |
| + 레이어 7층(첫층 0.02mm) | 0.785 | −21% | 커버리지 100% |
| + 후류 박스(0.625mm, 20d) | **1.257** | +26% | 준2D 효과(스팬 3D 미발달)로 과대측 |

레이어+후류 정밀화로 오차 −50% → ±26% 대역. 실제 그물은 교차점이 3D 붕괴를
유발하므로 준2D 과대측은 완화될 것으로 예상.

## 2. getFaceOrder FATAL 의 근본 원인과 해법

- 원인: 그물실이 **주기(cyclic) 경계를 관통** → 레이어 절단(truncation)이
  위상 결합된 cyclic 면쌍의 순서를 깨뜨림(polyTopoChange::getFaceOrder).
  nBufferCellsNoExtrude 1 로도 재발.
- **해법: cyclic → cyclicAMI**(translational + separationVector + lowWeight
  Correction 0.2). AMI 는 기하 보간 결합이라 위상 재순서 경로를 타지 않음 →
  크래시 소멸. 주기 기하라 보간 오차 미미(질량 오차 0 확인).

## 3. 그물(교차점 포함)에서의 레이어 커버리지 튜닝

| 시도 | 설정 | 커버리지 |
|---|---|---|
| 1 | 7층·첫층 0.02mm·fa60·기본 품질 | 24.9% (교차점 탈락 연쇄) |
| 2 | + featureAngle 130 + relaxed | 0.3% (역효과 — fa130 이 절단 연쇄 확대) |
| **3(채택)** | **4층·첫층 0.03mm·exp1.3·fa60·relaxed·nRelaxedIter 20** | **72.9%** |

교훈: 가는 와이어+교차점 격자에서는 '얇은 스택 + 완화 품질'이 정답.
featureAngle 상향은 금물.

## 4. 최종 결과 (U=1.0, 90° 정면)

| 케이스 | Cd | 비고 |
|---|---|---|
| UC 무레이어 (수정 BC) | 0.681 | 종전 |
| **UC + AMI + 레이어(수동 검증)** | **1.0734 ± 0.0004** | 정식 수렴(t=920), +58% |
| **UC 템플릿 E2E(빌더+러너 운영 경로)** | **1.0108 ± 0.0044** | 드리프트 0%, 시각화 8/8 |
| FS 무레이어 | 1.055–1.058 | 격자 무감(−0.6%) |
| FS + 레이어 시도 | 커버리지 **0%** | 표면 셀 1.9mm(1.6셀/직경)로 압출 전량 탈락 |
| Løland 이론 Cd_twine(Sn0.2, 90°) | 1.43 | 상단 참조 |

**핵심 성과: UC(1.01–1.07) ↔ FS(1.055) 가 ~4% 이내로 수렴** — 종전 35% 격차가
경계층 해상으로 해소됨. 절대값은 스크린 이론(1.43) 대비 아직 ~25% 낮을 수 있음
(미커버 교차점 27% + 4층 스택 한계) — 참조 시 유의.

## 5. 템플릿 반영 (운영 기본값 변경)

- `openfoam/unit_cell`: 측면 cyclic→**cyclicAMI**(0/* 전 필드 + blockMeshDict,
  separationVector 는 빌더 `_patch_blockMesh` 가 셀 크기로 자동 치환),
  snappy **addLayers true**(4층·첫층 3e-5m·exp1.3) + meshQuality **relaxed**.
- `openfoam/full_structure`: addLayers **false 유지** — 표면 해상도(1.9mm)가
  레이어 전제 미충족(커버리지 0%), FS 값은 격자 무감으로 이미 정합. 사유 주석화.
- `modules/cfd_manager.py`: separationVector 자동 치환 추가. cyclic 감지
  (`"cyclic" in boundary`)는 cyclicAMI 도 포함 → 직렬 snappy 경로 유지.

## 6. 영향·후속

- **앞으로의 UC 해석은 자동으로 경계층 격자 사용** — 절대 Cd 가 무레이어 대비
  ~1.5배 상향된 값으로 산출됨(예: A90 0.68→1.01). 기존 CSV 값과 혼용 금지.
- 잔여 개선 여지: 교차점 커버리지(27% 미커버), FS 레이어(표면 레벨 5+ 필요,
  셀 수 급증), 다각도(A45 등) 레이어 검증.
- 검증 케이스: bench_cyl_layers(2), v11uc_layers_A90, v11fs_layers_A90,
  v11_template_e2e (results/ 하위, 미추적).
