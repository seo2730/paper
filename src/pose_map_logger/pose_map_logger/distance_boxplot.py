#!/home/seo2730/duck_paper/bin/python
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

def extract_robot_number(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else float('inf')

def plot_robot_distance_boxplot(folder_path, save_filename='robot_distance_boxplot.png'):
    robot_data = {}

    print("\n📊 Analyzing total travel distances per robot:")

    for fname in os.listdir(folder_path):
        if fname.endswith('_distance.csv'):
            robot_name = fname.replace('_distance.csv', '')
            path = os.path.join(folder_path, fname)
            distances = pd.read_csv(path, header=None).squeeze().values

            mean_d = np.mean(distances)
            std_d = np.std(distances)

            print(f"- {robot_name}: mean = {mean_d:.2f} m, std = {std_d:.2f} m, experiments = {len(distances)}")

            robot_data[robot_name] = distances

    if not robot_data:
        print("❌ No '_distance.csv' files found in the specified folder.")
        return

    sorted_robot_names = sorted(robot_data.keys(), key=extract_robot_number)
    sorted_data = [robot_data[name] for name in sorted_robot_names]

    plt.figure(figsize=(10, 6))
    plt.boxplot(sorted_data, labels=sorted_robot_names, patch_artist=True,
                boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='none'))

    # for i, dists in enumerate(sorted_data):
    #     x = [i+1]*len(dists)
    #     plt.scatter(x, dists, color='red', zorder=3)

    plt.title('Total Travel Distance per Robot (Boxplot)')
    plt.ylabel('Travel Distance (m)')
    plt.ylim(5, 65)  # Y축 고정 범위 설정
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f'✅ Boxplot saved: {save_filename}')
    # plt.show()

def main():
    try:
        parser = argparse.ArgumentParser(description='Visualize robot travel distances as boxplot')
        parser.add_argument('--folder', type=str, required=True, help='Path to folder with distance CSVs')
        parser.add_argument('--output', type=str, default='robot_distance_boxplot.png', help='Filename to save the PNG image')
        args = parser.parse_args()

        plot_robot_distance_boxplot(args.folder, save_filename=args.output)

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user. Exiting gracefully.")
        sys.exit(0)

if __name__ == '__main__':
    main()
