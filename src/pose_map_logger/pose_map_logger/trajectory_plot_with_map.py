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

    h, w = grid.shape
    res = meta['resolution']
    ox, oy = meta['origin_x'], meta['origin_y']

    # 지도 extent를 실제 좌표계(m 단위)에 맞게 설정
    extent = [ox, ox + w * res, oy, oy + h * res]
    plt.imshow(map_img, cmap='gray', origin='lower', extent=extent)

    # 로봇 파일 수집
    robot_files = sorted([
        fname for fname in os.listdir(folder_path)
        if fname.endswith('.csv') and 'robot' in fname
    ])

    # 로봇 수만큼 고유 색상 지정
    cmap = plt.get_cmap('tab20', max(len(robot_files), 1))

    for idx, fname in enumerate(robot_files):
        robot_name = fname.replace('.csv', '')
        data = load_pose_data(os.path.join(folder_path, fname))

        x_world = data[:, 0]
        y_world = data[:, 1]

        color = cmap(idx)
        plt.plot(x_world, y_world, label=robot_name, color=color, linewidth=2, alpha=0.95)

        # 시작 위치 원으로 강조
        plt.scatter(
            x_world[0], y_world[0],
            s=80, c=[color], marker='o',
            edgecolors='black', linewidths=1.2,
            zorder=5, label=None
        )

        # 마지막 위치에 총 이동거리 표시
        dist = compute_total_distance(data)
        # plt.text(
        #     x_world[-1], y_world[-1],
        #     f'{dist:.2f}m', fontsize=9, color=color,
        #     ha='left', va='bottom'
        # )

        print(f'{robot_name} 총 이동 거리: {dist:.2f} m')

    # plt.title('Robot Trajectories over Map')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-1.5, 31.5) # -1.5, 35 -1.5, 26
    plt.ylim(-1.5, 31.5) # -1.5, 35  7, 31.5
    plt.xlabel('X (meters)', fontsize=20)
    plt.ylabel('Y (meters)', fontsize=20)
    plt.legend(loc='upper right', fontsize=18, framealpha=0.9)
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
