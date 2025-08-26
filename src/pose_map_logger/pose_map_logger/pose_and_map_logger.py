#!/home/seo2730/duck_paper/bin/python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import csv
import os
import numpy as np
import argparse

class PoseAndMapLogger(Node):
    def __init__(self, save_dir_name='실험결과'):
        super().__init__('pose_and_map_logger')
        
        self.pose_topics = [
            '/my111/posestamped',
            '/my112/posestamped',
            '/my119/posestamped',
            '/my120/posestamped',
            '/my221/posestamped',
            '/my228/posestamped',
            '/my001/posestamped',
            '/my005/posestamped',
            '/my006/posestamped',
            '/my117/posestamped',
            '/my229/posestamped',
            '/my230/posestamped',
        ]
        self.map_topic1 = '/my111/group_map_0_1'
        self.map_topic2 = '/my001/group_map_0_1'

        self.files = {}
        os.makedirs(save_dir_name, exist_ok=True)
        self.save_dir = save_dir_name

        for topic in self.pose_topics:
            fname = os.path.join(save_dir_name, topic.strip('/').replace('/', '_') + '.csv')
            f = open(fname, 'w', newline='')
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z'])
            self.files[topic] = (f, writer)
            self.create_subscription(PoseStamped, topic, self.pose_callback_factory(topic), 10)

        self.map_counter1 = 0
        self.map_counter2 = 0
        self.map_every_n = 10  # N개 메시지마다 저장
        self.create_subscription(OccupancyGrid, self.map_topic1, self.map_callback1, 10)
        self.create_subscription(OccupancyGrid, self.map_topic2, self.map_callback2, 10)

    def pose_callback_factory(self, topic):
        def callback(msg):
            pos = msg.pose.position
            self.files[topic][1].writerow([pos.x, pos.y, pos.z])
        return callback

    def map_callback1(self, msg):
        self.map_counter1 += 1
        if self.map_counter1 % self.map_every_n != 0:
            return

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin.position
        data = list(msg.data)
        
        grid = np.array(data, dtype=int).reshape((height, width))
        map_fname = f'map_group1_{self.map_counter1:04d}.csv'
        map_path = os.path.join(self.save_dir, map_fname)

        with open(map_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['width', width])
            writer.writerow(['height', height])
            writer.writerow(['resolution', resolution])
            writer.writerow(['origin_x', origin.x])
            writer.writerow(['origin_y', origin.y])
            writer.writerow(['origin_z', origin.z])
            writer.writerow([])
            writer.writerow(['grid'])
            for row in grid:
                writer.writerow(row)

        self.get_logger().info(f'🗺️ 지도 저장: {map_path}')
    
    def map_callback2(self, msg):
        self.map_counter2 += 1
        if self.map_counter2 % self.map_every_n != 0:
            return

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin.position
        data = list(msg.data)
        
        grid = np.array(data, dtype=int).reshape((height, width))
        map_fname = f'map_group2_{self.map_counter2:04d}.csv'
        map_path = os.path.join(self.save_dir, map_fname)

        with open(map_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['width', width])
            writer.writerow(['height', height])
            writer.writerow(['resolution', resolution])
            writer.writerow(['origin_x', origin.x])
            writer.writerow(['origin_y', origin.y])
            writer.writerow(['origin_z', origin.z])
            writer.writerow([])
            writer.writerow(['grid'])
            for row in grid:
                writer.writerow(row)

        self.get_logger().info(f'🗺️ 지도 저장: {map_path}')

    def destroy_node(self):
        for f, _ in self.files.values():
            f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    logger = PoseAndMapLogger(save_dir_name='map2/existing_algorithm/map1_existing_ex4')
    try:
        rclpy.spin(logger)
    except KeyboardInterrupt:
        logger.get_logger().info('종료 요청됨')
    finally:
        logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()