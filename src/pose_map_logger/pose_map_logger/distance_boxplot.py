#!/home/seo2730/duck_paper/bin/python
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_robot_distance_boxplot(folder_path, save_filename='robot_distance_boxplot.png'):
    labels = []
    data = []

    print("\n📊 로봇별 총 이동 거리 분석:")

    for fname in os.listdir(folder_path):
        if fname.endswith('_distance.csv'):
            robot_name = fname.replace('_distance.csv', '')
            path = os.path.join(folder_path, fname)
            distances = pd.read_csv(path, header=None).squeeze().values

            mean_d = np.mean(distances)
            std_d = np.std(distances)

            print(f"- {robot_name}: 평균={mean_d:.2f}m, 표준편차={std_d:.2f}m, 실험 수={len(distances)}")

            labels.append(robot_name)
            data.append(distances)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightgreen'))

    for i, dists in enumerate(data):
        x = [i+1]*len(dists)
        plt.scatter(x, dists, color='red', zorder=3)

    plt.title('로봇별 실험 총 이동 거리 Boxplot')
    plt.ylabel('이동 거리 (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f'✅ Boxplot 저장됨: {save_filename}')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='로봇 거리 Boxplot 시각화')
    parser.add_argument('--folder', type=str, required=True, help='거리 CSV들이 들어 있는 폴더 경로')
    parser.add_argument('--output', type=str, default='robot_distance_boxplot.png', help='저장할 PNG 파일 이름')
    args = parser.parse_args()

    plot_robot_distance_boxplot(args.folder, save_filename=args.output)

if __name__ == '__main__':
    main()