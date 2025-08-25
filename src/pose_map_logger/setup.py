#!/home/seo2730/duck_paper/bin/python
from setuptools import setup

package_name = 'pose_map_logger'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[],
    zip_safe=True,
    maintainer='seo2730',
    maintainer_email='seo2730@naver.com',
    description='Logs Pose and Map data, and provides visualization utilities for mobile robots.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pose_and_map_logger = pose_map_logger.pose_and_map_logger:main',
            'trajectory_plot_with_map = pose_map_logger.trajectory_plot_with_map:main',
            'distance_boxplot = pose_map_logger.distance_boxplot:main'
        ],
    },
)
