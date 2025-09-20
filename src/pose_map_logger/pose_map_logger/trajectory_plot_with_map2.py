#!/home/seo2730/duck_paper/bin/python
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    CSV 포맷 가정:
      resolution,0.05
      origin_x,-10.0
      origin_y,-10.0
      grid
      0,0,100, ...
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
# Map merging
# ----------------------------
def merge_maps(grid1, meta1, grid2, meta2):
    """
    해상도 동일해야 병합. (다르면 재표본화 필요 -> 본 코드에선 assert)
    서로 다른 origin/크기 지원: 월드 좌표계에서 하나의 큰 캔버스로 정렬 후 병합.
    병합 규칙(ROS 점유도 가정):
      - -1 = unknown
      - 0 = free
      - 100 = occupied
      - occupied(>=50)가 하나라도 있으면 occupied 우선
      - 둘 다 known이고 점유 없으면 free(0)
      - 둘 중 하나만 known이면 known 값
    """
    res1 = meta1['resolution']
    res2 = meta2['resolution']
    assert abs(res1 - res2) < 1e-12, "두 지도 resolution이 달라 병합할 수 없습니다."

    res = res1
    # 각 맵의 월드 범위 계산
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

    # unknown으로 초기화
    merged = -1 * np.ones((H, W), dtype=int)

    def paste(src_grid, src_meta):
        ox, oy = src_meta['origin_x'], src_meta['origin_y']
        x0 = int(np.floor((ox - x_min) / res))
        y0 = int(np.floor((oy - y_min) / res))
        h, w = src_grid.shape

        tgt = merged[y0:y0+h, x0:x0+w]
        src = src_grid

        a = tgt
        b = src

        only_a_known = (a != -1) & (b == -1)
        only_b_known = (a == -1) & (b != -1)
        both_known   = (a != -1) & (b != -1)

        result = a.copy()
        result[only_a_known] = a[only_a_known]
        result[only_b_known] = b[only_b_known]

        if np.any(both_known):
            a_known = a[both_known]
            b_known = b[both_known]
            occupied = (a_known >= 50) | (b_known >= 50)
            merged_vals = np.where(occupied, 100, 0)
            result[both_known] = merged_vals

        tgt[:, :] = result

    paste(grid1, meta1)
    paste(grid2, meta2)

    merged_meta = {
        'resolution': res,
        'origin_x' : x_min,
        'origin_y' : y_min
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

    # 시각화용 변환
    map_img = np.where(grid == -1, 127, 255 - grid)

    H, W = grid.shape
    res = meta['resolution']
    ox, oy = meta['origin_x'], meta['origin_y']

    # extent: 실제 월드(m) 좌표계
    extent = [ox, ox + W * res, oy, oy + H * res]
    plt.imshow(map_img, cmap='gray', origin='lower', extent=extent)

    # 로봇 파일 수집
    robot_files = sorted([
        fname for fname in os.listdir(folder_path)
        if fname.endswith('.csv') and 'posestamped' in fname
    ])

    cmap = plt.get_cmap('tab20', max(len(robot_files), 1))

    for idx, fname in enumerate(robot_files):
        robot_name = fname.replace('.csv', '')
        data = load_pose_data(os.path.join(folder_path, fname))

        x_world = data[:, 0]
        y_world = data[:, 1]

        color = cmap(idx)
        plt.plot(x_world, y_world, label=robot_name, color=color, linewidth=2, alpha=0.95)

        # 시작 위치 원으로 강조 (크기 = 80 고정)
        plt.scatter(
            x_world[0], y_world[0],
            s=80, c=[color], marker='o',
            edgecolors='black', linewidths=1.2, zorder=5, label=None
        )

        dist = compute_total_distance(data)
        plt.text(
            x_world[-1], y_world[-1],
            f'{dist:.2f}m', fontsize=9, color=color,
            ha='left', va='bottom'
        )

        print(f'{robot_name} 총 이동 거리: {dist:.2f} m')

    plt.title('Robot Trajectories over Map (merged)' if map_csv_path2 else 'Robot Trajectories over Map')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    # 범례 글씨 크기 20 고정
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
