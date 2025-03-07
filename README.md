# 机器学习辅助催化剂筛选
智能催化材料计算设计平台
<img width="1122" alt="image" src="https://github.com/user-attachments/assets/f673a9bd-c443-40a3-8c14-0e7b300b72dd" />

交叉领域：机器学习 × 计算化学 × 工业催化
技术标签：可解释性AI 多目标优化 知识增强图神经网络 高通量筛选

## 🌟 核心研究亮点
#### 单原子催化剂ML-DFT协同设计

​项目内容:《Achieving Pareto Optimality of Nitrogen Reduction Reaction Pathways Based on Explainable Machine Learning and First-Principles Calculations》本项目针对氮还原反应（NRR）双质子化路径，提出基于可解释机器学习（XAI）与高通量第一性原理计算的协同策略，筛选高效单原子电催化剂（SACs）；结合开源数据库与集成XGBoost、LightGBM、GBR与MLPs模型算法构建多模型融合框架，直接预测吉布斯  自由能（RMSE <0.2 eV），突破传统DFT计算效率瓶颈；结合SHAP（Shapley Additive Explanations）全局/局部解释性分析，揭示“Radius difference”为关键描述符，首次将催化火山理论融入ML模型，实现催化剂活性与稳定性的双目标优化。
#### 知识增强型图注意力网络（KAGAT）
<img width="1314" alt="image" src="https://github.com/user-attachments/assets/e4da4b5c-56e3-4050-9b6e-90ca484d5973" />


​架构创新：
项目内容: 本研究提出一种新型知识增强型图注意力网络架构（Knowledge-Augmented Graph Attention Network, KAGAT），通过设计双通道门控自适应机制实现化学领域先验知识与深度表征学习的动态融合。该模型构建了包含多头异构注意力子层的双路信息处理通道：结构感知通道采用改进的图注意力网络（GATv2），通过可微分边权重组机制捕捉分子图结构中的动态邻域关联；利用关系型图卷积提取官能团特性、电子效应等化学先验特征，创新性地设计了跨模态门控融合单元（Cross-modal Gating Unit, CMGU），通过可学习参数矩阵动态调节结构特征与知识特征的贡献权重本模型在分子性质预测任务中F1值提升8.2-12.7%，显著优于传统GAT及知识蒸馏方法，为化学智能计算提供了兼具数据驱动学习与领域知识解释性的新型范式。
#### ML-MC协同沸石催化剂设计

​工业导向：面向碳捕集与封存（CCUS）的催化剂筛选
​技术突破：
建立230种沸石构效数据库，贝叶斯优化指导千级参数空间搜索
XGBoost/MLP预测模型达到R²=0.93-0.95，锁定框架体积与最大腔径为核心描述符
筛选出5种高稳定性过渡金属掺杂方案（吸附能优化+负载量提升）
## 🏆 工程实践与学术产出

### 关键竞赛

​全国大学生化工设计大赛（二等奖）​ | 团队负责人
创新设计反应精馏隔壁塔，​能耗降低25%
开发ASPEN ACM膜反应器多物理场耦合模型
### 学术论文

#### 第一作者工作：《Machine Learning Guided Zeolite Catalyst Optimization》
期刊：MOLECULES (JCR Q2, 预计2025年6月提交)  

​第二作者工作：《CO2 Hydrogenation to Olefins: A Thermodynamic Study via ML》
状态：在投（导师通讯作者）
#### 🔍 模型可解释性案例

通过ML模型反演发现的描述符与催化火山理论一致性验证
根据注意力机制权重参数判断贡献

