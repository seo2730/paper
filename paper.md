# 실내 탐색 임무 수행을 위한 응집도 기반 유전 알고리즘 이종 그룹 편제 및 임무 할당 기법 (가제목)
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
Multi-Robot System(다중 로봇 시스템), Mission Task Allocation(임무 할당), Genetic Algorithm(유전 알고리즘), Coalition Formation(그룹 편제)

# 서론
 다중 로봇 시스템(Multi-Robot System, MRS)은 단일 로봇 시스템(Single-Robot System, SRS)의 자원 한계를 극복하기 위한 연구로 단일 로봇들이 각각 네트워크로 연결되어 다수의 로봇들로 이루어진 시스템이다. SRS는 단일 로봇이 모든 작업을 수행하는데 있어서 작업 처리 속도와 처리량에 한계가 있어 여러 로봇들이 협력하여 작업을 분담함으로써 효율성을 향상 시켰다. 특히 복잡한 작업 환경일수록 단일 로봇으로 처리하기 오래 걸릴 수 있지만 다수의 로봇들이 협업을 통해 복잡한 작업 환경을 극복할 수 있다. 게다가 SRS에서 단일 로봇이 고장난 경우 전체 시스템에 심각한 영향을 끼치지만 MRS는 일부 로봇이 고장 나더라도 나머지 로봇으로 작업을 이어갈 수 있어 MRS이 SRS보다 강건하고 안정한 시스템이다.[1]
 MRS는 통신 네트워크 기술이 발전함으로써 MRS에 대한 연구가 활발히 이루어지고 있다. 제어, 작업 할당, 통신, 경로 계획, 인공지능 등 다양한 기술[1-5] 분야에서 연구가 되어가고 있으며 산업에서는 재난, 물류, 탐사, 의료, 국방 등 다양한 분야[6-10]에 적용하는 중이다. 특히 국방 분야에서는 러시아의 우크라이나 침공 이후 무인기의 실전 투입률이 급진적으로 증가함으로써 MRS 기반 무기체계의 운용 개념의 중요성이 대두되고 있다. 하지만 운용자 입장에서는 단일 로봇을 운용했던 것보다 다수의 로봇들로 이루어진 무기체계를 운용하는 것이 상대적으로 복잡하고 어려울 수가 있다. 수행해야할 임무가 여러 개 있을 때 운용자가 수동적으로 여러 로봇들에게 임무를 할당하는 것은 비효율적이므로 운용자의 편의성을 위해 다중 로봇 임무 할당(Multi-robot task allocation, MRTA) 연구가 국내외 제안되고 있다.[11-13] MRTA는 3가지 기준인 로봇 능력, 과업 할당, 할당 시기로 구성된 분류 체계가 있다.[11] 첫째, 로봇 능력은 단일 과업 로봇(Single-Task Robots, ST)과 다중 과업 로봇(Multi-Task Robots, MT)으로 나뉜다. ST는 각 로봇이 한 번에 하나의 과업만 수행할 수 있으며, MT는 여러 과업을 동시에 수행할 수 있다. 둘째, 과업 할당은 단일 로봇 과업(Single-Robot Tasks, SR)과 다중 로봇 과업(Multi-Robot Tasks, MR)으로 구분된다. SR은 단일 로봇이 독립적으로 수행하는 과업이고, MR은 여러 로봇의 협력이 필요한 과업이다. 마지막으로, 할당 시기는 즉시 할당(Instantaneous Assignment, IA)과 시간 확장 할당(Time-Extended Assignment, TA)으로 나뉜다. IA는 과업이 즉각적으로 할당되며, TA는 장기적인 계획에 따라 할당된다. 본 논문에서는 다수 소형 로봇들로 이루어져 있어 여러 임무를 동시에 수행하기에는 성능의 한계가 있어 하나의 임무를 서로 협력하여 수행하는 ST과 MR를 채택했으며 운용자가 여러 임무 영역을 주었을 때 가까운 거리 순으로 할당 계획을 넣으므로 TA 특징이 있다. 즉 본 논문의 임무 할당 기법은 ST-MR-TA인 기법이다.
 다수의 로봇을 단일 탐색 임무에 동시에 할당할 경우, 전체 탐색 완료 시간을 단축시킬 수 있다. 그러나 제한된 임무 영역에 과도한 수의 로봇이 집중적으로 투입되면, 작업 할당의 공간적 불균형으로 인해 시스템 전반의 운용 효율성이 저하될 수 있다. 이러한 국지적 과밀 배치는 자원의 비효율적 활용뿐만 아니라, 오히려 임무 수행 시간의 증가로 이어질 가능성이 있다. 아울러, 로봇들이 연속적인 탐색 임무를 수행하는 과정에서 이전 임무로 인해 로봇의 상태(예: 배터리 잔량, 모터, 센서 등)가 변할 수 있으며, 이는 후속 임무 수행에 영향을 미칠 수 있다. 특히, 로봇들이 제한된 에너지 자원을 바탕으로 다수의 탐색 영역을 순차적으로 수행해야 하는 시나리오에서는, 탐색 순서와 경로에 따른 에너지 소모가 전체 임무의 성공 여부에 중요한 영향을 미친다. 예를 들어, 모든 로봇이 동일한 우선순위에 따라 일괄적으로 탐색 임무를 수행할 경우, 초기 탐색에서의 과도한 에너지 소비로 인해 일부 로봇이 후속 임무를 수행하지 못하게 되는 문제가 발생할 수 있다. 이러한 상황은 자원의 비효율적 운용을 초래할 뿐만 아니라, 시스템의 신뢰성과 지속 가능성을 저해하는 요인이 된다.
 이러한 문제에서 본 논문은 MRTA을 효과적으로 활용하는 방안으로 유전 알고리즘 기반 이종 그룹 편제 및 할당을 제안한다. 여기서 이종은 각 로봇들의 종류가 다른 것으로 본 논문에서는 무인지상차량(Unmanned Ground Vehicle, UGV) 2종류와 무인항공기(Unmanned Aerial Vehicle, UAV) 1종류를 활용한다. 최근 다양한 무인기가 개발됨으로써 이종을 고려한 임무계획 연구가 제안되고 있다.[12, 14] [12]는 이종을 고려한 임무 할당 논문으로 임무에 대한 성능 요구사항과 로봇의 성능을 활용한 선형계획법 알고리즘을 제안했다. 선형계획법은 연산은 빠르지만 복잡한 문제에서는 유연성이 부족하며 할당된 자원의 총 수 최소화라는 단일 목적함수와 다르게 다목적 최적화에 적용하기 어려운 부분이 있다. [14]는 이종을 고려한 그룹 편제한 논문으로 로봇의 성능을 고려하여 그룹을 편제했지만 UAV 종류만 다르게 했다. 본 논문은 하나의 임무에 대해 최대 성능을 낼 수 있으면서 최소 수량이라는 다목적 함수를 최적화기 위해 유전 알고리즘을 채택했다. 기존 알고리즘과 다르게 제시한 알고리즘은 

# 본론

# 시뮬레이션 결과

# 결론

# Reference
[1] Yan Z, Jouandeau N, Cherif AA. A Survey and Analysis of Multi-Robot Coordination. International Journal of Advanced Robotic Systems. 2013;10(12). doi:10.5772/57313
<br>
[2] K. M. Al-Aubidy, M. M. Ali and A. M. Derbas, "Multi-robot task scheduling and routing using neuro-fuzzy control," 2015 IEEE 12th International Multi-Conference on Systems, Signals & Devices (SSD15), Mahdia, Tunisia, 2015, pp. 1-6, doi: 10.1109/SSD.2015.7348097.
<br>
[3] Y. Huang, Y. Zhang and H. Xiao, "Multi-robot system task allocation mechanism for smart factory," 2019 IEEE 8th Joint International Information Technology and Artificial Intelligence Conference (ITAIC), Chongqing, China, 2019, pp. 587-591, doi: 10.1109/ITAIC.2019.8785546.
<br>
[4] Anton Andreychuk, Konstantin Yakovlev, Pavel Surynek, Dor Atzmon, Roni Stern, "Multi-agent pathfinding with continuous time", Artificial Intelligence, Volume 305, 2022, 103662, ISSN 0004-3702, https://doi.org/10.1016/j.artint.2022.103662.
<br>
[5] D. Silveria, K. Cabral and S. Givigi, "Scalable Swarm Control Using Deep Reinforcement Learning," 2025 IEEE International systems Conference (SysCon), Montreal, QC, Canada, 2025, pp. 1-8, doi: 10.1109/SysCon64521.2025.11014655. keywords: {Training;Target tracking;Navigation;Scalability;Surveillance;Pipelines;Neural networks;Deep reinforcement learning;Control systems;Multi-agent systems;Swarm control;multi-agent system;reinforcement learning},
<br>
[6] D. S. Drew, “Multi-agent systems for search and rescue applications,” Curr. Robot. Rep., vol. 2, pp. 189–200, 2021.
<br>
[7] R. N. Darmanin and M. K. Bugeja, "A review on multi-robot systems categorised by application domain," 2017 25th Mediterranean Conference on Control and Automation (MED), Valletta, Malta, 2017, pp. 701-706, doi: 10.1109/MED.2017.7984200. 
<br>
[8] M. J. Schuster et al., “The ARCHES space-analogue demonstration mission: Towards heterogeneous teams of autonomous robots for collaborative scientific sampling in planetary exploration,” IEEE Robot. Autom. Lett., vol. 5, pp. 5315–5322, Oct. 2020.
<br>
[9]  G. P. Das, T. M. McGinnity, S. A. Coleman, and L. Behera, “A distributed task allocation algorithm for a multi-robot system in healthcare facilities,” J. Intell. Robot. Syst., vol. 80, pp. 33–58, 2015.
<br>
[10] Z. Zhou, J. Liu and J. Yu, "A Survey of Underwater Multi-Robot Systems," in IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 1, pp. 1-18, January 2022, doi: 10.1109/JAS.2021.1004269.
keywords: {Oceans;Sea measurements;Unmanned underwater vehicles;Control systems;Multi-robot systems;Task analysis;Robots;Cooperation;formation control;multi-robot systems (MRS);taxonomy;underwater robots;underwater tasks},
<br>
[11] Gerkey BP, Matarić MJ. A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems. The International Journal of Robotics Research. 2004;23(9):939-954. doi:10.1177/0278364904045564
<br>
[12] Y. Lee, H. Kim, W. Park, and C. Kim, “Efficient Task-Resource Matchmaking Technique for Multiple/Heterogeneous Unmanned Combat Systems,” Journal of the Korea Institute of Military Science and Technology, vol. 26, no. 2. The Korea Institute of Military Science and Technology, pp. 188–196, 05-Apr-2023.
<br>
[13] Dai, W., Lu, H., Xiao, J. et al. Multi-Robot Dynamic Task Allocation for Exploration and Destruction. J Intell Robot Syst 98, 455–479 (2020). https://doi.org/10.1007/s10846-019-01081-3
<br>
[14] N. Qi, Z. Huang, F. Zhou, Q. Shi, Q. Wu and M. Xiao, "A Task-Driven Sequential Overlapping Coalition Formation Game for Resource Allocation in Heterogeneous UAV Networks," in IEEE Transactions on Mobile Computing, vol. 22, no. 8, pp. 4439-4455, 1 Aug. 2023, doi: 10.1109/TMC.2022.3165965.
keywords: {Task analysis;Games;Schedules;Resource management;Autonomous aerial vehicles;Costs;Reconnaissance;Unmanned aerial vehicle;overlapping coalition formation game;task and resource allocation;sequential task execution schedule},
<br>
[15] C. Shin, B.-M. Jeong, D. Suh, S. Shim, J. Kim, and H.-L. Choi, “Frontier Exploration and Task Allocation-based Cooperative Mapping Algorithm for Multi-robot System,” Journal of the Korea Institute of Military Science and Technology, vol. 28, no. 2. The Korea Institute of Military Science and Technology, pp. 217–223, 05-Apr-2025.
