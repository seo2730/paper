#!/home/seo2730/duck_paper/bin/python
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_pose_data(csv_path):
    df = pd.read_csv(csv_path)
    return df[['x', 'y', 'z']].values

def compute_total_distance(positions):
    diffs = np.diff(positions, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)

def load_map_csv(map_csv_path):
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
    grid = np.array([[int(x) for x in line.strip().split(',')] for line in grid_lines])
    return grid, meta

def plot_robot_trajectories_with_map(folder_path, map_csv_path, save_filename='robot_trajectories_with_map.png'):
    plt.figure(figsize=(10, 10))

    # 지도 시각화
    grid, meta = load_map_csv(map_csv_path)
    map_img = np.where(grid == -1, 127, 255 - grid)  # unknown=127, occupied=0, free=255
    plt.imshow(map_img, cmap='gray', origin='lower')

    # --- [추가] 로봇 파일 목록 수집 & 정렬(안정적 색/범례 순서) ---
    robot_files = sorted([
        fname for fname in os.listdir(folder_path)
        if fname.endswith('.csv') and 'posestamped' in fname
    ])

    # --- [추가] 로봇 수만큼 분리된 색을 갖는 컬러맵 준비 ---
    cmap = plt.get_cmap('tab20', max(len(robot_files), 1))  # len==0 방어

    # 로봇 궤적
    for idx, fname in enumerate(robot_files):
        robot_name = fname.replace('.csv', '')
        data = load_pose_data(os.path.join(folder_path, fname))

        # 월드 좌표 -> 맵 셀 좌표 변환
        x_cells = data[:, 0] / meta['resolution'] - meta['origin_x'] / meta['resolution']
        y_cells = data[:, 1] / meta['resolution'] - meta['origin_y'] / meta['resolution']

        # --- [변경] 고유 색 사용 ---
        color = cmap(idx)

        plt.plot(
            x_cells, y_cells,
            label=robot_name,
            color=color,
            linewidth=2, alpha=0.95
        )

        dist = compute_total_distance(data)
        plt.text(
            x_cells[-1], y_cells[-1],
            f'{dist:.2f}m', fontsize=9, color=color,
            ha='left', va='bottom'
        )

        print(f'{robot_name} 총 이동 거리: {dist:.2f} m')

    plt.title('Robot Trajectories over Map')
    plt.xlabel('Map X (cell)')
    plt.ylabel('Map Y (cell)')
    # 범례가 많아질 때도 보기 좋게
    plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=200)
    print(f'✅ 지도+궤적 그래프 저장됨: {save_filename}')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='궤적 + 맵 시각화')
    parser.add_argument('--folder', type=str, required=True, help='Pose CSV들이 있는 폴더 경로')
    parser.add_argument('--map', type=str, required=True, help='사용할 map CSV 파일 경로')
    parser.add_argument('--output', type=str, default='robot_trajectories_with_map.png', help='저장할 PNG 파일 이름')
    args = parser.parse_args()

    plot_robot_trajectories_with_map(args.folder, args.map, save_filename=args.output)

if __name__ == '__main__':
    main()
