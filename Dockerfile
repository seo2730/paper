FROM ros:foxy

# 1. 필수 패키지 업데이트 및 Python3 pip 설치
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# 2. 필요한 Python 패키지 설치
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib

# 3. 워크스페이스 디렉토리 준비
WORKDIR /root/duck_project
RUN git clone -b master https://github.com/seo2730/paper.git

# 4. 
WORKDIR /root/duck_project/paper
RUN . /opt/ros/foxy/setup.sh && \
    colcon build --symlink-install

# 5. 기본 실행환경 설정
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
RUN echo "source /root/duck_project/paper/install/setup.bash" >> ~/.bashrc

# 6. 런타임 워크스페이스 설정
ENV ROS_DOMAIN_ID=7
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
WORKDIR /root/ros2_ws
CMD ["bash"]
