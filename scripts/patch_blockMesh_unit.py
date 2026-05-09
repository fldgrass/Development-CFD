#!/usr/bin/env python3
"""
patch_blockMesh_unit.py
단위 셀 크기에 맞게 blockMeshDict 자동 수정 헬퍼
사용법: python3 patch_blockMesh_unit.py <blockMeshDict경로> <셀크기_mm>
"""
import sys
import re
from pathlib import Path

def patch(path_str: str, cell_mm: float):
    path    = Path(path_str)
    half    = cell_mm / 2.0
    depth   = half * 5.0
    cells_xy = max(10, int(20 * cell_mm / 20.0))
    cells_z  = max(50, int(100 * cell_mm / 20.0))

    text = path.read_text()
    # 꼭짓점 좌표 교체
    replacements = {
        r'\(-10 -10 -50\)': f'({-half:.1f} {-half:.1f} {-depth:.1f})',
        r'\( 10 -10 -50\)': f'({half:.1f} {-half:.1f} {-depth:.1f})',
        r'\( 10  10 -50\)': f'({half:.1f} {half:.1f} {-depth:.1f})',
        r'\(-10  10 -50\)': f'({-half:.1f} {half:.1f} {-depth:.1f})',
        r'\(-10 -10  50\)': f'({-half:.1f} {-half:.1f} {depth:.1f})',
        r'\( 10 -10  50\)': f'({half:.1f} {-half:.1f} {depth:.1f})',
        r'\( 10  10  50\)': f'({half:.1f} {half:.1f} {depth:.1f})',
        r'\(-10  10  50\)': f'({-half:.1f} {half:.1f} {depth:.1f})',
        r'\(20 20 100\)':   f'({cells_xy} {cells_xy} {cells_z})',
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    path.write_text(text)
    print(f"blockMeshDict 패치 완료: half={half}mm, depth={depth}mm, "
          f"cells=({cells_xy}×{cells_xy}×{cells_z})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: patch_blockMesh_unit.py <경로> <셀크기mm>")
        sys.exit(1)
    patch(sys.argv[1], float(sys.argv[2]))
