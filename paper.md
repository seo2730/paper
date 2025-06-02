# 실내 탐색 임무 수행을 위한 응집도 기반 이종 군집 유전 알고리즘 편제 및 임무 할당 기법 (가제목)
응집도를 고려한 유전 알고리즘 기반 이종 군집 편제 및 정찰 임무 할당 방법
이종 군집의 응집도 기반 편제 및 정찰 임무 할당을 위한 유전 알고리즘 기법

# Abstract
최근 군집 로봇 시스템은 탐색, 구조, 국방 분야 등 다양한 영역에서 많은 연구가 되어가고 있다. 
본 논문에서는 실내 탐색 영역의 군집 로봇 시스템을 집중적으로 다룬다.
다수의 로봇들을 지도 정보가 없는 실내 건물 전체 영역 중 하나의 탐색 영역에 전체 다 투입하면 해당 지역은 신속하게 탐색할 수 있다. 
그러나 일부 탐색 안된 영역들을 추가 탐색할 경우 각각 영역들을 모든 로봇들이 하나씩 순서대로 탐색하느라 시간이 더 걸릴 수 있다. 
즉 전체 실내 지역을 탐색하는데 있어서 모든 로봇들의 투입이 시간이 더 오래 걸릴 수 있다.
설령 실내 건물의 전체 영역을 하나의 탐색 영역을 해도 모든 로봇들이 각각 탐색한 곳에 대해 정보를 공유하느라 통신이 오래 걸려 영역을 나누어서 탐색한 것보다 오래 걸릴 수 있다.
실내 지도 정보가 없는 미탐색 지역에서 하나의 로봇보다는 다수의 로봇을 투입시켜 넓은 실내를 탐색하는게 더 효율적이며 다수의 로봇들을 하나의 탐사 지역에 한꺼번에 투입하기 보다는 탐사 지역 넓이에 맞게 개수를 선정하여 하나의 그룹을 편제하는 것이 더 효율적이다.
그러므로 실내 탐색 임무 수행을 위한 응집도 기반 이종 군집 유전 알고리즘 편제 및 할당 기법을 제안한다.
제안한 알고리즘은 유전 알고리즘으로 최적의 조합을 찾는 방식을 응용하여 최적의 군집 편제 및 할당을 한다.
또한 각 로봇들의 위치을 고려한 응집도를 고려하여 효율적인 탐색 임무 수행을 한다.
시뮬레이션을 통해 군집 편제를 고려한 제안한 알고리즘과 군집 편제를 고려하지 않는 할당 알고리즘을 비교하여 효율적인 탐색 임무 수행을 보여준다. 

### Keyword
Multi-Robot System(다중 로봇 시스템), Task Allocation(작업 할당), Genetic Algorithm(유전 알고리즘), Coalition Formation(그룹 편제)

# 서론
 Multi-robot system(MRS)은 Single-robot system(SRS)의 자원 한계를 극복하기 위한 연구로 단일 로봇들이 각각 네트워크로 연결되어 다수의 로봇들로 이루어진 시스템이다. 통신 기술이 발전함으로써 MRS에 대한 연구가 활발히 이루어지고 있다. SRS는 단일 로봇이 모든 작업을 수행하는데 있어서 작업 처리 속도와 처리량에 한계가 있어 여러 로봇이 협력하여 작업을 분담함으로써 효율성을 향상 시켰다. 특히 복잡한 작업 환경일수록 단일 로봇으로 처리하기 오래 걸릴 수 있는 작업이지만 다수의 로봇들이 협업을 통해 복잡한 작업 환경을 극복할 수 있다. 게다가 SRS에서 단일 로봇이 고장난 경우 전체 시스템에 심각한 영향을 끼치지만 MRS는 일부 로봇이 고장 나더라도 나머지 로봇으로 작업을 이어갈 수 있어 MRS이 SRS보다 견고하고 안정한 시스템이다.[1]
 MRS는 제어, 작업 할당, 통신, 경로 계획, 인공지능 등 다양한 기술[1-5] 분야에서 연구가 되어가고 있으며 산업에서는 재난, 물류, 탐사, 의료, 국방 등 다양한 분야[6-11]에 적용하는 중이다.

# 본론

# 시뮬레이션 결과

# 결론

# Reference
[1] Yan Z, Jouandeau N, Cherif AA. A Survey and Analysis of Multi-Robot Coordination. International Journal of Advanced Robotic Systems. 2013;10(12). doi:10.5772/57313
[2] K. M. Al-Aubidy, M. M. Ali and A. M. Derbas, "Multi-robot task scheduling and routing using neuro-fuzzy control," 2015 IEEE 12th International Multi-Conference on Systems, Signals & Devices (SSD15), Mahdia, Tunisia, 2015, pp. 1-6, doi: 10.1109/SSD.2015.7348097.
[3] Y. Huang, Y. Zhang and H. Xiao, "Multi-robot system task allocation mechanism for smart factory," 2019 IEEE 8th Joint International Information Technology and Artificial Intelligence Conference (ITAIC), Chongqing, China, 2019, pp. 587-591, doi: 10.1109/ITAIC.2019.8785546.
[4] Anton Andreychuk, Konstantin Yakovlev, Pavel Surynek, Dor Atzmon, Roni Stern, "Multi-agent pathfinding with continuous time", Artificial Intelligence, Volume 305, 2022, 103662, ISSN 0004-3702, https://doi.org/10.1016/j.artint.2022.103662.
[5] D. Silveria, K. Cabral and S. Givigi, "Scalable Swarm Control Using Deep Reinforcement Learning," 2025 IEEE International systems Conference (SysCon), Montreal, QC, Canada, 2025, pp. 1-8, doi: 10.1109/SysCon64521.2025.11014655. keywords: {Training;Target tracking;Navigation;Scalability;Surveillance;Pipelines;Neural networks;Deep reinforcement learning;Control systems;Multi-agent systems;Swarm control;multi-agent system;reinforcement learning},
[6] D. S. Drew, “Multi-agent systems for search and rescue applications,” Curr. Robot. Rep., vol. 2, pp. 189–200, 2021.
[7] R. N. Darmanin and M. K. Bugeja, "A review on multi-robot systems categorised by application domain," 2017 25th Mediterranean Conference on Control and Automation (MED), Valletta, Malta, 2017, pp. 701-706, doi: 10.1109/MED.2017.7984200. 
[8] M. J. Schuster et al., “The ARCHES space-analogue demonstration mission: Towards heterogeneous teams of autonomous robots for collaborative scientific sampling in planetary exploration,” IEEE Robot. Autom. Lett., vol. 5, pp. 5315–5322, Oct. 2020.
[9]  G. P. Das, T. M. McGinnity, S. A. Coleman, and L. Behera, “A distributed task allocation algorithm for a multi-robot system in healthcare facilities,” J. Intell. Robot. Syst., vol. 80, pp. 33–58, 2015.
[11] Z. Zhou, J. Liu and J. Yu, "A Survey of Underwater Multi-Robot Systems," in IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 1, pp. 1-18, January 2022, doi: 10.1109/JAS.2021.1004269.
keywords: {Oceans;Sea measurements;Unmanned underwater vehicles;Control systems;Multi-robot systems;Task analysis;Robots;Cooperation;formation control;multi-robot systems (MRS);taxonomy;underwater robots;underwater tasks},
[12]
