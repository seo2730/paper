#!/home/seo2730/duck_paper/bin/python
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ----------------------------
# 표시용 팔레트 (0=Free, 100=Occupied, -1=Unknown)
# ----------------------------
CLASS_CMAP = ListedColormap(["#FFFFFF", "#000000", "#7F7F7F"])  # 흰=free, 검=occ, 회=unknown
CLASS_NORM = BoundaryNorm([-1.5, -0.5, 50, 100.5], CLASS_CMAP.N)

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
      width,400
      height,400
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
            # width, height는 정수, 나머지는 float
            if key in ['width', 'height']:
                meta[key] = int(float(value))
            else:
                meta[key] = float(value)

    grid_lines = lines[grid_start:]
    grid = np.array([[int(x) for x in line.strip().split(',')] for line in grid_lines], dtype=int)
    return grid, meta

# ----------------------------
# 표시용 클래스 변환
# ----------------------------
def grid_to_display_classes(grid):
    """
    0=Free(흰색), 100=Occupied(검정), -1=Unknown(회색)
    """
    disp = np.full(grid.shape, 2, dtype=np.int8)  # Unknown=2
    known = (grid != -1)
    if not np.any(known):
        return disp
    occ_mask  = known & (grid >= 50)
    free_mask = known & (grid >= 0) & (grid < 50)
    disp[free_mask] = 0
    disp[occ_mask]  = 1
    return disp

# ----------------------------
# Map merging (origin_x, origin_y, width, height 고려, max 규칙)
# ----------------------------
def merge_maps(grid1, meta1, grid2, meta2):
    res1, res2 = meta1['resolution'], meta2['resolution']
    assert abs(res1 - res2) < 1e-12, "해상도가 다릅니다."
    res = res1

    # 지도1 범위
    h1, w1 = meta1['height'], meta1['width']
    x1_min, y1_min = meta1['origin_x'], meta1['origin_y']
    x1_max, y1_max = x1_min + w1 * res, y1_min + h1 * res

    # 지도2 범위
    h2, w2 = meta2['height'], meta2['width']
    x2_min, y2_min = meta2['origin_x'], meta2['origin_y']
    x2_max, y2_max = x2_min + w2 * res, y2_min + h2 * res

    # 전체 바운딩 박스
    x_min, x_max = min(x1_min, x2_min), max(x1_max, x2_max)
    y_min, y_max = min(y1_min, y2_min), max(y1_max, y2_max)
    W = int(np.ceil((x_max - x_min) / res))
    H = int(np.ceil((y_max - y_min) / res))

    merged = -1 * np.ones((H, W), dtype=int)

    def paste(src_grid, src_meta):
        h, w = src_meta['height'], src_meta['width']
        ox, oy = src_meta['origin_x'], src_meta['origin_y']
        x0 = int(np.floor((ox - x_min) / res))
        y0 = int(np.floor((oy - y_min) / res))
        return (x0, y0, w, h)

    # 지도1 복사
    x0, y0, w, h = paste(grid1, meta1)
    merged[y0:y0+h, x0:x0+w] = grid1

    # 지도2 병합
    x1, y1, w2, h2 = paste(grid2, meta2)
    sub = merged[y1:y1+h2, x1:x1+w2]
    a = sub
    b = grid2

    both_known = (a != -1) & (b != -1)
    only_a = (a != -1) & (b == -1)
    only_b = (a == -1) & (b != -1)

    out = a.copy()
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    out[both_known] = np.maximum(a[both_known], b[both_known])  # ✅ max 규칙

    merged[y1:y1+h2, x1:x1+w2] = out

    merged_meta = {
        'resolution': res,
        'origin_x': x_min,
        'origin_y': y_min,
        'width': W,
        'height': H
    }
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

    H, W = meta['height'], meta['width']
    res = meta['resolution']
    ox, oy = meta['origin_x'], meta['origin_y']
    extent = [ox, ox + W * res, oy, oy + H * res]

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
    parser = argparse.ArgumentParser(description='궤적 + 맵 시각화 (단일/병합 지도, max 규칙)')
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
