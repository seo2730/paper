#!/home/seo2730/duck_paper/bin/python
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ----------------------------
# 표시용 팔레트 (0=Free, 1=Occupied, 2=Unknown)
# ----------------------------
CLASS_CMAP = ListedColormap(["#FFFFFF", "#000000", "#7F7F7F"])  # 흰=free, 검=occ, 회=unknown
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CLASS_CMAP.N)

# ----------------------------
# Data loaders & utilities
# ----------------------------
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
      grid
      0,0,100,...
      ...
    """
    with open(map_csv_path, 'r') as f:
        lines = f.readlines()

    meta = {}
    grid_start = 0
    for i, line in enumerate(lines):
        if 'grid' in line:
            grid_start = i + 1
            break
        if ',' in line:
            key, value = line.strip().split(',')
            meta[key] = float(value)

    grid_lines = lines[grid_start:]
    grid = np.array([[int(x) for x in line.strip().split(',')] for line in grid_lines], dtype=int)
    return grid, meta

# ----------------------------
# 표시용 3-클래스 변환 (색 고정)
# ----------------------------
def grid_to_display_classes(grid):
    """
    0=Free(흰색), 1=Occupied(검정), 2=Unknown(회색)
    -1은 Unknown 고정. 임계 50 기준으로 Free/Occupied 분류.
    """
    disp = np.full(grid.shape, 2, dtype=np.int8)  # Unknown=2
    known = (grid != -1)
    if not np.any(known):
        return disp
    occ_mask  = known & (grid >= 50)
    free_mask = known & (grid < 50)
    disp[free_mask] = 0
    disp[occ_mask]  = 1
    return disp

# ----------------------------
# Map merging (unknown 대체 + 합집합 규칙)
# ----------------------------
def merge_maps(grid1, meta1, grid2, meta2):
    res1, res2 = meta1['resolution'], meta2['resolution']
    assert abs(res1 - res2) < 1e-12, "두 지도 resolution이 달라 병합할 수 없습니다."
    res = res1

    # 두 지도 범위
    h1, w1 = grid1.shape
    h2, w2 = grid2.shape
    x1_min, x1_max = meta1['origin_x'], meta1['origin_x'] + w1 * res
    y1_min, y1_max = meta1['origin_y'], meta1['origin_y'] + h1 * res
    x2_min, x2_max = meta2['origin_x'], meta2['origin_x'] + w2 * res
    y2_min, y2_max = meta2['origin_y'], meta2['origin_y'] + h2 * res

    # 전체 범위
    x_min, x_max = min(x1_min, x2_min), max(x1_max, x2_max)
    y_min, y_max = min(y1_min, y2_min), max(y1_max, y2_max)
    W = int(np.ceil((x_max - x_min) / res))
    H = int(np.ceil((y_max - y_min) / res))

    merged = -1 * np.ones((H, W), dtype=int)

    def place(src_grid, src_meta):
        ox, oy = src_meta['origin_x'], src_meta['origin_y']
        x0 = int(np.floor((ox - x_min) / res))
        y0 = int(np.floor((oy - y_min) / res))
        h, w = src_grid.shape
        return (y0, y0+h, x0, x0+w), src_grid

    (y10,y1h,x10,x1w), g1 = place(grid1, meta1)
    (y20,y2h,x20,x2w), g2 = place(grid2, meta2)

    # 지도1 전체 복사
    merged[y10:y1h, x10:x1w] = g1

    # 지도2 붙이기 (unknown 처리 포함)
    # 교차 영역
    dx0, dx1 = max(x10,x20), min(x1w,x2w)
    dy0, dy1 = max(y10,y20), min(y1h,y2h)
    if dx0 < dx1 and dy0 < dy1:
        a = merged[dy0:dy1, dx0:dx1]
        b = g2[(dy0-y20):(dy1-y20), (dx0-x20):(dx1-x20)]
        out = a.copy()

        both_known = (a != -1) & (b != -1)
        only_a     = (a != -1) & (b == -1)
        only_b     = (a == -1) & (b != -1)

        # 한쪽만 known → 그 값
        out[only_a] = a[only_a]
        out[only_b] = b[only_b]

        # 둘 다 known → 규칙 적용
        if np.any(both_known):
            occ = (a[both_known] >= 50) | (b[both_known] >= 50)
            out[both_known] = np.where(occ, 100, 0)

        merged[dy0:dy1, dx0:dx1] = out

    # 교차 외 영역에서 지도2 값이 있고 merged가 unknown이면 채워줌
    yslice = slice(max(0,y20), min(H,y2h))
    xslice = slice(max(0,x20), min(W,x2w))
    sub = merged[yslice, xslice]
    g2_crop = g2[(yslice.start-y20):(yslice.stop-y20),
                 (xslice.start-x20):(xslice.stop-x20)]
    mask = (sub == -1) & (g2_crop != -1)
    sub[mask] = g2_crop[mask]
    merged[yslice, xslice] = sub

    merged_meta = {'resolution': res, 'origin_x': x_min, 'origin_y': y_min}
    return merged, merged_meta

# ----------------------------
# Plotting
# ----------------------------
def plot_robot_trajectories_with_map(folder_path,
                                     map_csv_path,
                                     map_csv_path2=None,
                                     save_filename='robot_trajectories_with_map.png'):
    plt.figure(figsize=(10, 10))

    # 지도 로드(+병합)
    grid1, meta1 = load_map_csv(map_csv_path)
    if map_csv_path2:
        grid2, meta2 = load_map_csv(map_csv_path2)
        grid, meta = merge_maps(grid1, meta1, grid2, meta2)
    else:
        grid, meta = grid1, meta1

    H, W = grid.shape
    res = meta['resolution']
    ox, oy = meta['origin_x'], meta['origin_y']
    extent = [ox, ox + W * res, oy, oy + H * res]

    # 3-클래스 팔레트로 표시
    disp = grid_to_display_classes(grid)
    plt.imshow(disp, cmap=CLASS_CMAP, norm=CLASS_NORM,
               origin='lower', extent=extent)

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

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='궤적 + 맵 시각화 (단일/병합 지도)')
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
