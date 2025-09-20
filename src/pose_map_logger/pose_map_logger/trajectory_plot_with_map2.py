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
CLASS_CMAP = ListedColormap(["#FFFFFF", "#000000", "#7F7F7F"])
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
# 표시용 3-클래스 변환 (스케일 차이와 상관없이 색 고정)
# ----------------------------
def grid_to_display_classes(grid):
    """
    0=Free(흰색), 1=Occupied(검정), 2=Unknown(회색)
    -1은 Unknown 고정. 나머지는 임계 50 기준으로 Free/Occupied 분류.
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
# Map merging (덮어쓰기 방지: 위치 클리핑 + 합집합 규칙)
# ----------------------------
def merge_maps(grid1, meta1, grid2, meta2):
    """
    해상도 동일 필요. 서로 다른 origin/크기 지원.
    병합 규칙:
      - 하나라도 Occupied(>=50)이면 100
      - 그 외, 알려진 값이 하나라도 있으면 0(Free)
      - 둘 다 Unknown이면 -1
    + 붙여넣기 위치를 반드시 캔버스 내부로 '클리핑'하여 음수 인덱싱 래핑 방지
    """
    res1, res2 = meta1['resolution'], meta2['resolution']
    assert abs(res1 - res2) < 1e-12, "두 지도 resolution이 달라 병합할 수 없습니다."
    res = res1

    h1, w1 = grid1.shape
    h2, w2 = grid2.shape
    x1_min, x1_max = meta1['origin_x'], meta1['origin_x'] + w1 * res
    y1_min, y1_max = meta1['origin_y'], meta1['origin_y'] + h1 * res
    x2_min, x2_max = meta2['origin_x'], meta2['origin_x'] + w2 * res
    y2_min, y2_max = meta2['origin_y'], meta2['origin_y'] + h2 * res

    x_min, x_max = min(x1_min, x2_min), max(x1_max, x2_max)
    y_min, y_max = min(y1_min, y2_min), max(y1_max, y2_max)

    W = int(np.ceil((x_max - x_min) / res))
    H = int(np.ceil((y_max - y_min) / res))

    merged = -1 * np.ones((H, W), dtype=int)  # Unknown으로 초기화

    def paste_union(src_grid, src_meta):
        ox, oy = src_meta['origin_x'], src_meta['origin_y']
        x0 = int(np.floor((ox - x_min) / res))
        y0 = int(np.floor((oy - y_min) / res))
        h, w = src_grid.shape

        # ---- 클리핑 (음수/범위초과 방지) ----
        # 대상(캔버스) 좌표
        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(W, x0 + w)
        dy1 = min(H, y0 + h)

        # 아무 교집합이 없으면 skip
        if dx0 >= dx1 or dy0 >= dy1:
            return

        # 소스(원본) 좌표
        sx0 = dx0 - x0
        sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)

        dst = merged[dy0:dy1, dx0:dx1]
        src = src_grid[sy0:sy1, sx0:sx1]

        a = dst
        b = src

        a_unknown = (a == -1)
        b_unknown = (b == -1)

        a_occ = (~a_unknown) & (a >= 50)
        b_occ = (~b_unknown) & (b >= 50)

        occ = a_occ | b_occ
        known = (~a_unknown) | (~b_unknown)
        free = known & (~occ)

        out = -1 * np.ones_like(a)
        out[occ] = 100   # occupied
        out[free] = 0    # free

        dst[:, :] = out  # 규칙 결과만 기록 (덮어쓰기 아님)

    # 순서는 중요하지 않지만 두 번 모두 '합집합 규칙' + '클리핑'으로 안전
    paste_union(grid1, meta1)
    paste_union(grid2, meta2)

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

    # 3-클래스 고정 팔레트로 시각화(지도들 간 톤 일관성 확보)
    disp = grid_to_display_classes(grid)
    plt.imshow(disp, cmap=CLASS_CMAP, norm=CLASS_NORM,
               origin='lower', extent=extent)

    # 로봇 파일 수집(안정적 색/범례 순서를 위해 정렬)
    robot_files = sorted([
        fname for fname in os.listdir(folder_path)
        if fname.endswith('.csv') and 'posestamped' in fname
    ])

    # 로봇 수만큼 고유 색상
    cmap = plt.get_cmap('tab20', max(len(robot_files), 1))

    for idx, fname in enumerate(robot_files):
        robot_name = fname.replace('.csv', '')
        data = load_pose_data(os.path.join(folder_path, fname))

        x_world = data[:, 0]  # m
        y_world = data[:, 1]  # m

        color = cmap(idx)
        # 궤적
        plt.plot(x_world, y_world, label=robot_name, color=color, linewidth=2, alpha=0.95)

        # 시작 위치 원 강조 (s=80 고정)
        plt.scatter(x_world[0], y_world[0],
                    s=80, c=[color], marker='o',
                    edgecolors='black', linewidths=1.2,
                    zorder=5, label=None)

        # 마지막 위치에 총 이동거리 표시
        dist = compute_total_distance(data)
        plt.text(x_world[-1], y_world[-1],
                 f'{dist:.2f}m', fontsize=9, color=color,
                 ha='left', va='bottom')

        print(f'{robot_name} 총 이동 거리: {dist:.2f} m')

    plt.title('Robot Trajectories over Map (merged)' if map_csv_path2 else 'Robot Trajectories over Map')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(loc='upper right', fontsize=20, markerscale=1.3, framealpha=0.9)  # 범례 20 고정
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=200)
    print(f'✅ 지도+궤적 그래프 저장됨: {save_filename}')
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='궤적 + 맵 시각화 (단일/병합 지도, 안전 클리핑)')
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
