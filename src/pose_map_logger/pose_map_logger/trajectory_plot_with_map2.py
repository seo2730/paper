#!/home/seo2730/duck_paper/bin/python
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# =========================
# 표시 팔레트: 0=Free(흰), >0=Occ(검), -1=Unknown(회)
# =========================
CLASS_CMAP = ListedColormap(["#FFFFFF", "#000000", "#7F7F7F"])  # [free, occ, unknown]
# 값 자체로 분류하지 않고, 우리가 만든 클래스 인덱스(0,1,2)를 그릴 거라 norm은 의미상 placeholder
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CLASS_CMAP.N)

# =========================
# 로드/유틸
# =========================
def load_pose_data(csv_path):
    df = pd.read_csv(csv_path)
    return df[['x', 'y', 'z']].values

def compute_total_distance(positions):
    diffs = np.diff(positions, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.sum(dists))

def load_map_csv(map_csv_path):
    """
    CSV 예시:
      resolution,0.05
      origin_x,-10.0
      origin_y,-10.0
      width,400
      height,400
      grid
      0,0,5,0, ...
      ...
    """
    with open(map_csv_path, 'r') as f:
        lines = f.readlines()

    meta = {}
    grid_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == 'grid':
            grid_start = i + 1
            break
        if ',' in line:
            key, value = line.strip().split(',', 1)
            key = key.strip()
            value = value.strip()
            if key in ['width', 'height']:
                meta[key] = int(float(value))
            else:
                meta[key] = float(value)

    grid_lines = [ln.strip() for ln in lines[grid_start:] if ln.strip()]
    grid = np.array([[int(x) for x in row.split(',')] for row in grid_lines], dtype=int)

    # 안전 체크: shape가 meta와 일치해야 함 (행=height, 열=width)
    h_expected = meta.get('height', grid.shape[0])
    w_expected = meta.get('width',  grid.shape[1])
    assert grid.shape == (h_expected, w_expected), \
        f"grid shape {grid.shape} != (height,width)=({h_expected},{w_expected})"

    return grid, meta

def grid_to_display_classes(grid):
    """
    free=0, occupied>0, unknown=-1을 색상 클래스(0,1,2)로 변환
      0 -> free(흰), >0 -> occ(검), -1 -> unknown(회)
    """
    disp = np.full(grid.shape, 2, dtype=np.uint8)  # 기본 unknown(2)
    unknown = (grid == -1)
    free    = (grid == 0)
    occ     = (grid > 0)

    disp[free] = 0
    disp[occ]  = 1
    disp[unknown] = 2
    return disp

# =========================
# 병합 (원점/크기 기반 배치 + 클리핑 + unknown 대체 + overlap은 max)
# =========================
def merge_maps_max_rule(grid1, meta1, grid2, meta2):
    # 해상도 동일 필수
    res1 = float(meta1['resolution'])
    res2 = float(meta2['resolution'])
    assert abs(res1 - res2) < 1e-12, "resolution mismatch"
    res = res1

    # 각 지도 bbox (월드 좌표)
    def bbox(meta):
        w = int(meta['width']); h = int(meta['height'])
        ox = float(meta['origin_x']); oy = float(meta['origin_y'])
        return ox, oy, ox + w*res, oy + h*res, w, h

    x1min,y1min,x1max,y1max,w1,h1 = bbox(meta1)
    x2min,y2min,x2max,y2max,w2,h2 = bbox(meta2)

    # 통합 bbox 및 캔버스 크기
    xmin, ymin = min(x1min, x2min), min(y1min, y2min)
    xmax, ymax = max(x1max, x2max), max(y1max, y2max)
    W = int(np.ceil((xmax - xmin) / res))
    H = int(np.ceil((ymax - ymin) / res))

    merged = -1 * np.ones((H, W), dtype=int)

    # 통합 캔버스에서 시작 인덱스
    def start_index(meta):
        ox = float(meta['origin_x']); oy = float(meta['origin_y'])
        x0 = int(np.floor((ox - xmin) / res))
        y0 = int(np.floor((oy - ymin) / res))
        return x0, y0

    x0_1, y0_1 = start_index(meta1)
    x0_2, y0_2 = start_index(meta2)

    # 공통: 안전 클리핑 + 규칙 적용 paste
    def paste_into(dst, src, x0, y0):
        H, W = dst.shape
        h, w = src.shape

        # 캔버스 클리핑 (음수/초과 인덱싱 방지)
        dx0 = max(0, x0);           dy0 = max(0, y0)
        dx1 = min(W, x0 + w);       dy1 = min(H, y0 + h)
        if dx0 >= dx1 or dy0 >= dy1:
            return

        sx0 = dx0 - x0;  sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0);  sy1 = sy0 + (dy1 - dy0)

        a = dst[dy0:dy1, dx0:dx1]      # 기존
        b = src[sy0:sy1, sx0:sx1]      # 신규

        a_unknown = (a == -1)
        b_unknown = (b == -1)
        both_known = (~a_unknown) & (~b_unknown)

        out = a.copy()
        # 한쪽만 known이면 known 값으로 채움
        out[a_unknown & ~b_unknown] = b[a_unknown & ~b_unknown]
        out[~a_unknown & b_unknown] = a[~a_unknown & b_unknown]
        # 둘 다 known이면 max 선택
        out[both_known] = np.maximum(a[both_known], b[both_known])

        dst[dy0:dy1, dx0:dx1] = out

    # 지도1 먼저 반영, 그 다음 지도2를 같은 규칙으로 병합
    paste_into(merged, grid1, x0_1, y0_1)
    paste_into(merged, grid2, x0_2, y0_2)

    merged_meta = {
        'resolution': res,
        'origin_x': xmin,
        'origin_y': ymin,
        'width': W,
        'height': H
    }
    return merged, merged_meta

# =========================
# Plot
# =========================
def plot_robot_trajectories_with_map(folder_path,
                                     map_csv_path,
                                     map_csv_path2=None,
                                     save_filename='robot_trajectories_with_map.png'):
    plt.figure(figsize=(10, 10))

    # 지도 로드(+병합)
    grid1, meta1 = load_map_csv(map_csv_path)
    if map_csv_path2:
        grid2, meta2 = load_map_csv(map_csv_path2)
        grid, meta = merge_maps_max_rule(grid1, meta1, grid2, meta2)
    else:
        grid, meta = grid1, meta1

    H, W = int(meta['height']), int(meta['width'])
    res = float(meta['resolution'])
    ox, oy = float(meta['origin_x']), float(meta['origin_y'])
    extent = [ox, ox + W * res, oy, oy + H * res]

    # 지도 표시 (클래스 인덱스 기반)
    disp = grid_to_display_classes(grid)
    plt.imshow(disp, cmap=CLASS_CMAP, norm=CLASS_NORM, origin='lower', extent=extent)

    # 로봇 궤적
    robot_files = sorted([
        fname for fname in os.listdir(folder_path)
        if fname.endswith('.csv') and 'posestamped' in fname
    ])
    cmap = plt.get_cmap('tab20', max(len(robot_files), 1))

    for idx, fname in enumerate(robot_files):
        robot_name = fname.replace('.csv', '')
        data = load_pose_data(os.path.join(folder_path, fname))
        x_world, y_world = data[:, 0], data[:, 1]
        color = cmap(idx)

        plt.plot(x_world, y_world, label=robot_name, color=color, linewidth=2, alpha=0.95)
        # 시작점 원
        plt.scatter(x_world[0], y_world[0], s=80, c=[color], marker='o',
                    edgecolors='black', linewidths=1.2, zorder=5, label=None)

        dist = compute_total_distance(data)
        plt.text(x_world[-1], y_world[-1], f'{dist:.2f}m',
                 fontsize=9, color=color, ha='left', va='bottom')
        print(f'{robot_name} 총 이동 거리: {dist:.2f} m')

    plt.title('Robot Trajectories over Map (merged)' if map_csv_path2 else 'Robot Trajectories over Map')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(loc='upper right', fontsize=20, markerscale=1.3, framealpha=0.9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=200)
    print(f'✅ 지도+궤적 그래프 저장됨: {save_filename}')
    plt.show()

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description='궤적 + 맵 시각화 (단일/병합 지도, overlap=max, free=0/occ>0)')
    parser.add_argument('--folder', type=str, required=True, help='Pose CSV들이 있는 폴더 경로')
    parser.add_argument('--map', type=str, required=True, help='사용할 map CSV 파일 경로(필수)')
    parser.add_argument('--map2', type=str, default=None, help='병합할 두 번째 map CSV 파일 경로(선택)')
    parser.add_argument('--output', type=str, default='robot_trajectories_with_map.png', help='저장 PNG 파일 이름')
    args = parser.parse_args()

    plot_robot_trajectories_with_map(
        folder_path=args.folder,
        map_csv_path=args.map,
        map_csv_path2=args.map2,
        save_filename=args.output
    )

if __name__ == '__main__':
    main()
