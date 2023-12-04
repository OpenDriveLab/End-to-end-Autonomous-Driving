<div id="top">

# Paper Collection

We list key challenges from a wide span of candidate concerns, as well as trending methodologies.

- [Survey](#survey)
- [Language-guided Driving](#language-guided-driving)
- [World Model & Model-based RL](#world-model--model-based-rl)
- [Multi-sensor Fusion](#multi-sensor-fusion)
- [Multi-task Learning](#multi-task-learning)
- [Interpretability](#interpretability)
  - [Review](#review)
  - [Attention Visualization](#attention-visualization)
  - [Interpretable Tasks](#interpretable-tasks)
  - [Cost Learning](#cost-learning)
  - [Linguistic Explainability](#linguistic-explainability)
  - [Uncertainty Modeling](#uncertainty-modeling)
  - [Counterfactual Explanations and Causal Inference](#counterfactual-explanations-and-causal-inference)
- [Visual Abstraction / Representation Learning](#visual-abstraction--representation-learning)
- [Policy Distillation](#policy-distillation)
- [Causal Confusion](#causal-confusion)
- [Robustness](#robustness)
  - [Long-tailed Distribution](#long-tailed-distribution)
  - [Covariate Shift](#covariate-shift)
  - [Domain Adaptation](#domain-adaptation)
- [Affordance Learning](#affordance-learning)
- [BEV](#bev)
- [Transformer](#transformer)
- [V2V Cooperative](#v2v-cooperative)
- [Distributed RL](#distributed-rl)
- [Data-driven Simulation](#data-driven-simulation)
  - [Parameter Initialization](#parameter-initialization)
  - [Traffic Simulation](#traffic-simulation)
  - [Sensor Simulation](#sensor-simulation)


## Survey

<!-- <details><summary>(Click for details)</summary> -->

- End-to-End Autonomous Driving: Challenges and Frontiers [[arXiv2023]](https://arxiv.org/abs/2306.16927)

- Recent Advancements in End-to-End Autonomous Driving using Deep Learning: A Survey [[TIV2023]](https://ieeexplore.ieee.org/abstract/document/10258330)

- Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review [[arXiv2023]](https://arxiv.org/abs/2308.05731)

- Motion Planning for Autonomous Driving: The State of the Art and Future Perspectives [[TIV2023]](https://arxiv.org/abs/2303.09824)

- Imitation Learning: Progress, Taxonomies and Challenges [[TNNLS2022]](https://arxiv.org/abs/2106.12177)

- A Review of End-to-End Autonomous Driving in Urban Environments [[Access2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9832636)

- A Survey on Imitation Learning Techniques for End-to-End Autonomous Vehicles [[TITS2022]](https://arxiv.org/abs/2101.01993)

- Deep Reinforcement Learning for Autonomous Driving: A Survey [[TITS2021]](https://arxiv.org/abs/2002.00444)

- A Survey of Deep RL and IL for Autonomous Driving Policy Learning [[TITS2021]](https://arxiv.org/abs/2101.01993)

- A Survey of End-to-End Driving: Architectures and Training Methods [[TNNLS2020]](https://arxiv.org/abs/2003.06404)

- Learning to Drive by Imitation: An Overview of Deep Behavior Cloning Methods [[TIV2020]](https://ieeexplore.ieee.org/abstract/document/9117169)

- Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art [[book]](https://arxiv.org/abs/1704.05519)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>


## Language-guided Driving

<!-- <details><summary>(Click for details)</summary> -->

- LLM4Drive: A Survey of Large Language Models for Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2311.01043)

- A Language Agent for Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2311.10813)

- GPT-Driver: Learning to Drive with GPT [[arXiv2023]](https://arxiv.org/abs/2310.01415)

- DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model [[arXiv2023]](https://arxiv.org/abs/2310.01412)

- Ground then Navigate: Language-guided Navigation in Dynamic Scenes [[arXiv2022]](https://arxiv.org/abs/2209.11972)

- Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving [[arXiv2023]](https://browse.arxiv.org/abs/2310.01957)

- DOROTHIE: Spoken Dialogue for Handling Unexpected Situations in Interactive Autonomous Driving Agents
 [[EMNLP2022(Findings)]](https://arxiv.org/abs/2210.12511)[[code]](https://github.com/sled-group/DOROTHIE)

- LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action [[CoRL2022]](https://proceedings.mlr.press/v205/shah23b.html)

- Generating Landmark Navigation Instructions from Maps as a Graph-to-Text Problem [[ACL2021]](https://arxiv.org/abs/2012.15329)

- Advisable Learning for Self-Driving Vehicles by Internalizing Observation-to-Action Rules [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Advisable_Learning_for_Self-Driving_Vehicles_by_Internalizing_Observation-to-Action_Rules_CVPR_2020_paper.html)
    
- Conditional Driving from Natural Language Instructions [[CoRL2019]](https://arxiv.org/abs/1910.07615)
    
- Grounding Human-to-Vehicle Advice for Self-driving Vehicles [[CVPR2019]](https://arxiv.org/abs/1911.06978)[[Dataset]](https://usa.honda-ri.com/had)
    
- Talk to the Vehicle: Language Conditioned Autonomous Navigation of Self Driving Cars [[IROS2019]](https://ieeexplore.ieee.org/abstract/document/8967929)

- Talk2Car: Taking Control of Your Self-Driving Car [[EMNLP2019]](https://arxiv.org/abs/1909.10838)

- TOUCHDOWN: Natural Language Navigation and Spatial Reasoning in Visual Street Environments [[CVPR2019]](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_TOUCHDOWN_Natural_Language_Navigation_and_Spatial_Reasoning_in_Visual_Street_CVPR_2019_paper.html)

- Learning to Navigate in Cities Without a Map [[NeurIPS2018]](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e034fb6b66aacc1d48f445ddfb08da98-Abstract.html)[[Code]](https://github.com/deepmind/streetlearn)![](https://img.shields.io/github/stars/deepmind/streetlearn.svg?style=social&label=Star&maxAge=2592000)
  
<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>  

## World Model & Model-based RL

<!-- <details><summary>(Click for details)</summary> -->

- GAIA-1: A Generative World Model for Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2309.17080)

- ADriver-I: A General World Model for Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2311.13549)

- OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2311.16038)[[Code]](https://github.com/wzzheng/OccWorld)![](https://img.shields.io/github/stars/wzzheng/OccWorld.svg?style=social&label=Star&maxAge=2592000)

- DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2309.09777)[[Code]](https://github.com/JeffWang987/DriveDreamer)![](https://img.shields.io/github/stars/JeffWang987/DriveDreamer.svg?style=social&label=Star&maxAge=2592000)

- Uncertainty-Aware Model-Based Offline Reinforcement Learning for Automated Driving [[RAL2023]](https://ieeexplore.ieee.org/document/10015868)

- Model-Based Imitation Learning for Urban Driving [[NeurIPS2022)]](https://arxiv.org/abs/2210.07729)[[Code]](https://github.com/wayveai/mile.git)![](https://img.shields.io/github/stars/wayveai/mile.svg?style=social&label=Star&maxAge=2592000)

- Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models [[NeurIPS2022]](http://arxiv.org/pdf/2205.13817v3)[[Code]](https://github.com/panmt/Iso-Dream.git)![](https://img.shields.io/github/stars/panmt/Iso-Dream.svg?style=social&label=Star&maxAge=2592000)

- Enhance Sample Efficiency and Robustness of End-to-end Urban Autonomous Driving via Semantic Masked World Model [[NeurIPSWorkshop2022]](https://arxiv.org/abs/2210.04017)

- Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning [[ICML2022]](https://arxiv.org/abs/2207.10295)

- Interpretable End-to-End Urban Autonomous Driving With Latent Deep Reinforcement Learning [[TITS2022]](https://arxiv.org/abs/2001.08726)[[Code]](https://github.com/cjy1992/interp-e2e-driving.git)![](https://img.shields.io/github/stars/cjy1992/interp-e2e-driving.svg?style=social&label=Star&maxAge=2592000)

- Learning To Drive From a World on Rails [[ICCV2021]](http://arxiv.org/pdf/2105.00636v3)[[Code]](https://github.com/dotchen/WorldOnRails.git)![](https://img.shields.io/github/stars/dotchen/WorldOnRails.svg?style=social&label=Star&maxAge=2592000)

- Uncertainty-Aware Model-Based Reinforcement Learning: Methodology and Application in Autonomous Driving [[IV2022]](https://ieeexplore.ieee.org/abstract/document/9802913)

- UMBRELLA: Uncertainty-Aware Model-Based Offline Reinforcement Learning Leveraging Planning [[NeurIPSWorkshop2021]](https://arxiv.org/pdf/2111.11097.pdf)

- Deductive Reinforcement Learning for Visual Autonomous Urban Driving Navigation [[TNNLS2021]](https://ieeexplore.ieee.org/document/9537641)

- Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic [[ICLR2019]](https://arxiv.org/abs/1901.02705)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Multi-sensor Fusion

<!-- <details><summary>(Click for details)</summary> -->

- Hidden Biases of End-to-End Driving Models [[ICCV2023]](https://arxiv.org/abs/2306.07957)[[code]](https://github.com/autonomousvision/carla_garage)![](https://img.shields.io/github/stars/autonomousvision/carla_garage.svg?style=social&label=Star&maxAge=2592000)

- Learning to Drive Anywhere [[CoRL2023]](https://arxiv.org/abs/2309.12295)

- Think Twice before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2305.06242)[[Code]](https://github.com/OpenDriveLab/ThinkTwice)![](https://img.shields.io/github/stars/OpenDriveLab/ThinkTwice.svg?style=social&label=Star&maxAge=2592000)

- ReasonNet: End-to-End Driving with Temporal and Global Reasoning [[CVPR2023]](https://arxiv.org/abs/2305.10507)

- FusionAD: Multi-modality Fusion for Prediction and Planning Tasks of Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2308.01006)

- Enhance Sample Efficiency and Robustness of End-to-end Urban Autonomous Driving via Semantic Masked World Model [[NeurIPSWorkshop2022]](https://arxiv.org/abs/2210.04017)

- End-to-end Autonomous Driving with Semantic Depth Cloud Mapping and Multi-agent [[IV2022]](https://arxiv.org/abs/2204.05513)

- MMFN: Multi-Modal-Fusion-Net for End-to-End Driving [[IROS2022]](https://arxiv.org/abs/2207.00186)[[code]](https://github.com/Kin-Zhang/mmfn)![](https://img.shields.io/github/stars/Kin-Zhang/mmfn.svg?style=social&label=Star&maxAge=2592000)

- Interpretable End-to-End Urban Autonomous Driving With Latent Deep Reinforcement Learning [[TITS2022]](https://arxiv.org/abs/2001.08726)[[Code]](https://github.com/cjy1992/interp-e2e-driving.git)![](https://img.shields.io/github/stars/cjy1992/interp-e2e-driving.svg?style=social&label=Star&maxAge=2592000)

- Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer [[CoRL2022]](https://arxiv.org/abs/2207.14024)[[Code]](https://github.com/opendilab/InterFuser)![](https://img.shields.io/github/stars/opendilab/InterFuser.svg?style=social&label=Star&maxAge=2592000)

- Learning from All Vehicles [[CVPR2022]](http://arxiv.org/pdf/1709.04622v4)[[Code]](https://github.com/dotchen/LAV.git)![](https://img.shields.io/github/stars/dotchen/LAV.svg?style=social&label=Star&maxAge=2592000)

- TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving [[TPAMI2022]](https://arxiv.org/abs/2205.15997)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- Multi-Modal Fusion Transformer for End-to-End Autonomous Driving [[CVPR2021]](https://arxiv.org/abs/2104.09224)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- Carl-Lead: Lidar-based End-to-End Autonomous Driving with Contrastive Deep Reinforcement Learning [[arXiv2021]](https://arxiv.org/abs/2109.08473)

- Multi-modal Sensor Fusion-based Deep Neural Network for End-to-end Autonomous Driving with Scene Understanding [[IEEESJ2020]](https://arxiv.org/abs/2005.09202)
    
- Probabilistic End-to-End Vehicle Navigation in Complex Dynamic Environments With Multimodal Sensor Fusion [[RAL2020]](https://arxiv.org/abs/2005.01935)

- Multimodal End-to-End Autonomous Driving [[TITS2020]](https://ieeexplore.ieee.org/abstract/document/9165167)

- End-To-End Interpretable Neural Motion Planner [[CVPR2019]](https://openaccess.thecvf.com/content_CVPR_2019/html/Zeng_End-To-End_Interpretable_Neural_Motion_Planner_CVPR_2019_paper.html)

- Does Computer Vision Matter for Action? [[ScienceRobotics2019]](https://www.science.org/doi/abs/10.1126/scirobotics.aaw6661)

- End-To-End Multi-Modal Sensors Fusion System For Urban Automated Driving [[NeurIPSWorkshop2018]](https://openreview.net/forum?id=Byx4Xkqjcm)

- MultiNet: Multi-Modal Multi-Task Learning for Autonomous Driving [[WACV2019]](https://arxiv.org/abs/1709.05581)

- LiDAR-Video Driving Dataset: Learning Driving Policies Effectively [[CVPR2018]](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_LiDAR-Video_Driving_Dataset_CVPR_2018_paper.html)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Multi-task Learning

<!-- <details><summary>(Click for details)</summary> -->

- Planning-oriented Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2212.10156)[[Code]](https://github.com/OpenDriveLab/UniAD)![](https://img.shields.io/github/stars/OpenDriveLab/UniAD.svg?style=social&label=Star&maxAge=2592000)

- Think Twice before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2305.06242)[[code]](https://github.com/OpenDriveLab/ThinkTwice)![](https://img.shields.io/github/stars/OpenDriveLab/ThinkTwice.svg?style=social&label=Star&maxAge=2592000)

- Coaching a Teachable Student [[CVPR2023]](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Coaching_a_Teachable_Student_CVPR_2023_paper.html)

- ReasonNet: End-to-End Driving with Temporal and Global Reasoning [[CVPR2023]](https://arxiv.org/abs/2305.10507)

- Hidden Biases of End-to-End Driving Models [[ICCV2023]](https://arxiv.org/abs/2306.07957)[[code]](https://github.com/autonomousvision/carla_garage)![](https://img.shields.io/github/stars/autonomousvision/carla_garage.svg?style=social&label=Star&maxAge=2592000)

- VAD: Vectorized Scene Representation for Efficient Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2303.12077)[[code]](https://github.com/hustvl/VAD)![](https://img.shields.io/github/stars/hustvl/VAD.svg?style=social&label=Star&maxAge=2592000)

- TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving [[TPAMI2022]](https://arxiv.org/abs/2205.15997)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)
    
- Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline [[NeurIPS2022]](https://arxiv.org/abs/2206.08129) [[code]](https://github.com/OpenDriveLab/TCP)![](https://img.shields.io/github/stars/OpenDriveLab/TCP.svg?style=social&label=Star&maxAge=2592000)

- Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer [[CoRL2022]](https://arxiv.org/abs/2207.14024)[[Code]](https://github.com/opendilab/InterFuser)![](https://img.shields.io/github/stars/opendilab/InterFuser.svg?style=social&label=Star&maxAge=2592000)

- Learning from All Vehicles [[CVPR2022]](http://arxiv.org/pdf/1709.04622v4)[[Code]](https://github.com/dotchen/LAV.git)![](https://img.shields.io/github/stars/dotchen/LAV.svg?style=social&label=Star&maxAge=2592000)

- Multi-Task Learning With Attention for End-to-End Autonomous Driving [[CVPRWorkshop2021]](https://arxiv.org/abs/2104.10753)
    
- NEAT: Neural Attention Fields for End-to-End Autonomous Driving [[ICCV2021]](https://arxiv.org/abs/2109.04456)[[Code]](https://github.com/autonomousvision/neat.git)![](https://img.shields.io/github/stars/autonomousvision/neat.svg?style=social&label=Star&maxAge=2592000)

- SAM: Squeeze-and-Mimic Networks for Conditional Visual Driving Policy Learning [[CoRL2020]](https://arxiv.org/abs/1912.02973)[[Code]](https://github.com/twsq/sam-driving.git)![](https://img.shields.io/github/stars/twsq/sam-driving.svg?style=social&label=Star&maxAge=2592000)

- Urban Driving with Conditional Imitation Learning [[ICRA2020]](http://arxiv.org/pdf/1912.00177v2)
   
- Multi-modal Sensor Fusion-based Deep Neural Network for End-to-end Autonomous Driving with Scene Understanding [[IEEESJ2020]](https://arxiv.org/abs/2005.09202)

- Multi-task Learning with Future States for Vision-based Autonomous Driving [[ACCV2020]](https://openaccess.thecvf.com/content/ACCV2020/papers/Kim_Multi-task_Learning_with_Future_States_for_Vision-based_Autonomous_Driving_ACCV_2020_paper.pdf)

- Learning to Steer by Mimicking Features from Heterogeneous Auxiliary Networks [[AAAI2019]](http://arxiv.org/pdf/1811.02759v1)[[Code]](https://github.com/cardwing/Codes-for-Steering-Control.git)![](https://img.shields.io/github/stars/cardwing/Codes-for-Steering-Control.svg?style=social&label=Star&maxAge=2592000)
    
- MultiNet: Multi-Modal Multi-Task Learning for Autonomous Driving [[WACV2019]](https://arxiv.org/abs/1709.05581)
    
- Intentnet: Learning to Predict Intention from Raw Sensor Data [[CoRL2018]](https://arxiv.org/abs/2101.07907)
    
- Rethinking Self-driving: Multi-task Knowledge for Better Generalization and Accident Explanation Ability [[arXiv2018]](https://arxiv.org/abs/1809.11100)[[Code]](https://github.com/jackspp/rethinking-self-driving.git)![](https://img.shields.io/github/stars/jackspp/rethinking-self-driving.svg?style=social&label=Star&maxAge=2592000)
   
- Learning End-to-end Autonomous Driving using Guided Auxiliary Supervision [[ICVGIP2018]](https://arxiv.org/abs/1808.10393)

- End-to-end Learning of Driving Models from Large-scale Video Datasets [[CVPR2017]](https://arxiv.org/abs/1612.01079)[[Code]](https://github.com/gy20073/BDD_Driving_Model.git)![](https://img.shields.io/github/stars/gy20073/BDD_Driving_Model.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>  

## Interpretability

### Review

- Explainability of deep vision-based autonomous driving systems: Review and challenges [[IJCV2022]](https://arxiv.org/abs/2101.05307)

### Attention Visualization

<!-- <details><summary>(Click for details)</summary> -->

- Scaling Self-Supervised End-to-End Driving with Multi-View Attention Learning [[arxiv2023]](https://arxiv.org/abs/2302.03198)

- PlanT: Explainable Planning Transformers via Object-Level Representations [[CoRL2022]](https://arxiv.org/abs/2210.14222)[[Code]](https://github.com/autonomousvision/plant)![](https://img.shields.io/github/stars/autonomousvision/plant.svg?style=social&label=Star&maxAge=2592000)

- MMFN: Multi-Modal-Fusion-Net for End-to-End Driving [[IROS2022]](https://arxiv.org/abs/2207.00186)[[code]](https://github.com/Kin-Zhang/mmfn)![](https://img.shields.io/github/stars/Kin-Zhang/mmfn.svg?style=social&label=Star&maxAge=2592000)

- TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving [[TPAMI2022]](https://arxiv.org/abs/2205.15997)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- Multi-Modal Fusion Transformer for End-to-End Autonomous Driving [[CVPR2021]](https://arxiv.org/abs/2104.09224)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- Multi-Task Learning With Attention for End-to-End Autonomous Driving [[CVPRWorkshop2021]](https://arxiv.org/abs/2104.10753)

- NEAT: Neural Attention Fields for End-to-End Autonomous Driving [[ICCV2021]](https://arxiv.org/abs/2109.04456)[[Code]](https://github.com/autonomousvision/neat.git)![](https://img.shields.io/github/stars/autonomousvision/neat.svg?style=social&label=Star&maxAge=2592000)

- Explaining Autonomous Driving by Learning End-to-End Visual Attention [[CVPRWorkshop2020]](https://openaccess.thecvf.com/content_CVPRW_2020/html/w20/Cultrera_Explaining_Autonomous_Driving_by_Learning_End-to-End_Visual_Attention_CVPRW_2020_paper.html)

- Visual Explanation by Attention Branch Network for End-to-end Learning-based Self-driving [[IV2019]](https://ieeexplore.ieee.org/abstract/document/8813900)

- Deep Object-Centric Policies for Autonomous Driving [[ICRA2019]](https://ieeexplore.ieee.org/abstract/document/8794224)

- Textual Explanations for Self-Driving Vehicles [[ECCV2018]](https://openaccess.thecvf.com/content_ECCV_2018/html/Jinkyu_Kim_Textual_Explanations_for_ECCV_2018_paper.html)[[Code]](https://github.com/JinkyuKimUCB/explainable-deep-driving)![](https://img.shields.io/github/stars/JinkyuKimUCB/explainable-deep-driving.svg?style=social&label=Star&maxAge=2592000)

- Learning End-to-end Autonomous Driving using Guided Auxiliary Supervision [[ICVGIP2018]](https://arxiv.org/abs/1808.10393)

- Interpretable Learning for Self-Driving Cars by Visualizing Causal Attention [[ICCV2017]](https://openaccess.thecvf.com/content_iccv_2017/html/Kim_Interpretable_Learning_for_ICCV_2017_paper.html)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
### Interpretable Tasks

<!-- <details><summary>(Click for details)</summary> -->

- Planning-oriented Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2212.10156)[[Code]](https://github.com/OpenDriveLab/UniAD)![](https://img.shields.io/github/stars/OpenDriveLab/UniAD.svg?style=social&label=Star&maxAge=2592000)

- Hidden Biases of End-to-End Driving Models [[ICCV2023]](https://arxiv.org/abs/2306.07957)[[code]](https://github.com/autonomousvision/carla_garage)![](https://img.shields.io/github/stars/autonomousvision/carla_garage.svg?style=social&label=Star&maxAge=2592000)

- VAD: Vectorized Scene Representation for Efficient Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2303.12077)[[code]](https://github.com/hustvl/VAD)![](https://img.shields.io/github/stars/hustvl/VAD.svg?style=social&label=Star&maxAge=2592000)

- Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer [[CoRL2022]](https://arxiv.org/abs/2207.14024)[[Code]](https://github.com/opendilab/InterFuser)![](https://img.shields.io/github/stars/opendilab/InterFuser.svg?style=social&label=Star&maxAge=2592000)

- TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving [[TPAMI2022]](https://arxiv.org/abs/2205.15997)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- Learning from All Vehicles [[CVPR2022]](http://arxiv.org/pdf/1709.04622v4)[[Code]](https://github.com/dotchen/LAV.git)![](https://img.shields.io/github/stars/dotchen/LAV.svg?style=social&label=Star&maxAge=2592000)

- Ground then Navigate: Language-guided Navigation in Dynamic Scenes [[arXiv2022]](https://arxiv.org/abs/2209.11972)

- NEAT: Neural Attention Fields for End-to-End Autonomous Driving [[ICCV2021]](https://arxiv.org/abs/2109.04456)[[Code]](https://github.com/autonomousvision/neat.git)![](https://img.shields.io/github/stars/autonomousvision/neat.svg?style=social&label=Star&maxAge=2592000)

- Multi-Task Learning With Attention for End-to-End Autonomous Driving [[CVPRWorkshop2021]](https://arxiv.org/abs/2104.10753)

- Urban Driving with Conditional Imitation Learning [[ICRA2020]](http://arxiv.org/pdf/1912.00177v2)

- Using Eye Gaze to Enhance Generalization of Imitation Networks to Unseen Environments [[TNNLS2020]](https://www.ram-lab.com/papers/2020/liu2020tnnls.pdf)

- Multi-modal Sensor Fusion-based Deep Neural Network for End-to-end Autonomous Driving with Scene Understanding [[IEEESJ2020]](https://arxiv.org/abs/2005.09202)

- Rethinking Self-driving: Multi-task Knowledge for Better Generalization and Accident Explanation Ability [[arXiv2018]](https://arxiv.org/abs/1809.11100)[[Code]](https://github.com/jackspp/rethinking-self-driving.git)![](https://img.shields.io/github/stars/jackspp/rethinking-self-driving.svg?style=social&label=Star&maxAge=2592000)

- Learning End-to-end Autonomous Driving using Guided Auxiliary Supervision [[ICVGIP2018]](https://arxiv.org/abs/1808.10393)

- End-to-end Learning of Driving Models from Large-scale Video Datasets [[CVPR2017]](https://arxiv.org/abs/1612.01079)[[Code]](https://github.com/gy20073/BDD_Driving_Model.git)![](https://img.shields.io/github/stars/gy20073/BDD_Driving_Model.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
### Cost Learning

<!-- <details><summary>(Click for details)</summary> -->

- ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning [[ECCV2022]](https://arxiv.org/abs/2207.07601)[[Code]](https://github.com/OpenDriveLab/ST-P3)![](https://img.shields.io/github/stars/OpenDriveLab/ST-P3.svg?style=social&label=Star&maxAge=2592000)

- Differentiable Raycasting for Self-Supervised Occupancy Forecasting [[ECCV2022]](https://arxiv.org/abs/2210.01917)[[Code]](https://github.com/tarashakhurana/emergent-occ-forecasting.git)![](https://img.shields.io/github/stars/tarashakhurana/emergent-occ-forecasting.svg?style=social&label=Star&maxAge=2592000)

- MP3: A Unified Model To Map, Perceive, Predict and Plan [[CVPR2021]](https://arxiv.org/abs/2101.06806)

- Safe Local Motion Planning With Self-Supervised Freespace Forecasting [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Safe_Local_Motion_Planning_With_Self-Supervised_Freespace_Forecasting_CVPR_2021_paper.html)

- LookOut: Diverse Multi-Future Prediction and Planning for Self-Driving [[ICCV2021]](https://arxiv.org/abs/2101.06547)

- DSDNet: Deep Structured Self-driving Network [[ECCV2020]](https://arxiv.org/abs/2008.06041)

- Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations [[ECCV2020]](https://arxiv.org/abs/2008.05930)

- End-To-End Interpretable Neural Motion Planner [[CVPR2019]](https://openaccess.thecvf.com/content_CVPR_2019/html/Zeng_End-To-End_Interpretable_Neural_Motion_Planner_CVPR_2019_paper.html)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
### Linguistic Explainability

<!-- <details><summary>(Click for details)</summary> -->

- ADAPT: Action-aware Driving Caption Transformer [[ICRA2023]](https://arxiv.org/abs/2302.00673)[[Code]](https://github.com/jxbbb/ADAPT)![](https://img.shields.io/github/stars/jxbbb/ADAPT.svg?style=social&label=Star&maxAge=2592000)

- Driving Behavior Explanation with Multi-level Fusion [[PR2022]](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005975)[[Code]](https://github.com/valeoai/BEEF)![](https://img.shields.io/github/stars/valeoai/BEEF.svg?style=social&label=Star&maxAge=2592000)

- Explainable Object-Induced Action Decision for Autonomous Vehicles [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Explainable_Object-Induced_Action_Decision_for_Autonomous_Vehicles_CVPR_2020_paper.html)

- Textual Explanations for Self-Driving Vehicles [[ECCV2018]](https://openaccess.thecvf.com/content_ECCV_2018/html/Jinkyu_Kim_Textual_Explanations_for_ECCV_2018_paper.html)[[Code]](https://github.com/JinkyuKimUCB/explainable-deep-driving)![](https://img.shields.io/github/stars/JinkyuKimUCB/explainable-deep-driving.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
### Uncertainty Modeling

<!-- <details><summary>(Click for details)</summary> -->

- UAP-BEV: Uncertainty Aware Planning using Bird's Eye View generated from Surround Monocular Images [[CASE2023]](https://arxiv.org/abs/2306.04939)[[code]](https://github.com/Vikr-182/UAP-BEV)![](https://img.shields.io/github/stars/Vikr-182/UAP-BEV.svg?style=social&label=Star&maxAge=2592000)

- Probabilistic End-to-End Vehicle Navigation in Complex Dynamic Environments With Multimodal Sensor Fusion [[RAL2020]](https://arxiv.org/abs/2005.01935)

- Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts? [[ICML2020]](https://arxiv.org/abs/2006.14911)[[Code]](https://github.com/OATML/oatomobile.git)![](https://img.shields.io/github/stars/OATML/oatomobile.svg?style=social&label=Star&maxAge=2592000)

- VTGNet: A Vision-Based Trajectory Generation Network for Autonomous Vehicles in Urban Environments [[TIV2020]](https://arxiv.org/abs/2004.12591)[[Code]](https://github.com/caipeide/VTGNet.git)![](https://img.shields.io/github/stars/caipeide/VTGNet.svg?style=social&label=Star&maxAge=2592000)

- Visual-based Autonomous Driving Deployment from a Stochastic and Uncertainty-aware Perspective [[IROS2019]](https://ieeexplore.ieee.org/abstract/document/8968307)

- Evaluating Uncertainty Quantification in End-to-End Autonomous Driving Control [[arXiv2018]](https://arxiv.org/abs/1811.06817)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

### Counterfactual Explanations and Causal Inference

<!-- <details><summary>(Click for details)</summary> -->
 
- OCTET: Object-aware Counterfactual Explanation [[CVPR2023]](https://arxiv.org/abs/2211.12380)[[Code]](https://github.com/valeoai/OCTET.git)![](https://img.shields.io/github/stars/valeoai/OCTET.svg?style=social&label=Star&maxAge=2592000)

- STEEX: Steering Counterfactual Explanations with Semantics [[ECCV2022]](https://arxiv.org/abs/2111.09094)[[Code]](https://github.com/valeoai/STEEX.git)![](https://img.shields.io/github/stars/valeoai/STEEX.svg?style=social&label=Star&maxAge=2592000)

- Who Make Drivers Stop? Towards Driver-centric Risk Assessment: Risk Object Identification via Causal Inference [[IROS2020]](https://arxiv.org/abs/2003.02425)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
## Visual Abstraction / Representation Learning

<!-- <details><summary>(Click for details)</summary> -->

- Scene as Occupancy [[ICCV2023]](https://arxiv.org/abs/2306.02851)[[Code]](https://github.com/OpenDriveLab/OccNet)![](https://img.shields.io/github/stars/OpenDriveLab/OccNet?style=social&label=Star)

- DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2308.00398)[[Code]](https://github.com/OpenDriveLab/DriveAdapter)![](https://img.shields.io/github/stars/OpenDriveLab/DriveAdapter?style=social&label=Star)

- Policy Pre-training for Autonomous Driving via Self-supervised Geometric Modeling [[ICLR2023]](https://openreview.net/forum?id=X5SUR7g2vVw)[[Code]](https://github.com/OpenDriveLab/PPGeo)![](https://img.shields.io/github/stars/OpenDriveLab/PPGeo.svg?style=social&label=Star&maxAge=2592000)

- Pre-Trained Image Encoder for Generalizable Visual Reinforcement Learning [[NeurIPS2022]](https://openreview.net/forum?id=FQtku8rkp3)

- Task-Induced Representation Learning [[ICLR2022]](https://arxiv.org/abs/2204.11827)[[Code]](https://github.com/clvrai/tarp)![](https://img.shields.io/github/stars/clvrai/tarp.svg?style=social&label=Star&maxAge=2592000)

- Learning Generalizable Representations for Reinforcement Learning via Adaptive Meta-learner of Behavioral Similarities [[ICLR2022]](https://arxiv.org/abs/2212.13088)[[Code]](https://github.com/jianda-chen/AMBS.git)![](https://img.shields.io/github/stars/jianda-chen/AMBS.svg?style=social&label=Star&maxAge=2592000)

- Learning to Drive by Watching YouTube Videos: Action-Conditioned Contrastive Policy Pretraining [[ECCV2022]](https://arxiv.org/abs/2204.02393)[[Code]](https://github.com/metadriverse/ACO)![](https://img.shields.io/github/stars/metadriverse/ACO.svg?style=social&label=Star&maxAge=2592000)

- Segmented Encoding for Sim2Real of RL-based End-to-End Autonomous Driving [[IV2022]](https://ieeexplore.ieee.org/abstract/document/9827374)

- GRI: General Reinforced Imitation and its Application to Vision-Based Autonomous Driving [[arXiv2021]](https://arxiv.org/abs/2111.08575)

- Latent Attention Augmentation for Robust Autonomous Driving Policies [[IROS2021]](https://ieeexplore.ieee.org/abstract/document/9636449)

- Multi-Task Long-Range Urban Driving Based on Hierarchical Planning and Reinforcement Learning [[ITSC2021]](https://ieeexplore.ieee.org/abstract/document/9564705)

- Carl-Lead: Lidar-based End-to-End Autonomous Driving with Contrastive Deep Reinforcement Learning [[arXiv2021]](https://arxiv.org/abs/2109.08473)

- A Versatile and Efficient Reinforcement Learning Framework for Autonomous Driving [[arxiv2021]](https://arxiv.org/abs/2110.11573)

- Deductive Reinforcement Learning for Visual Autonomous Urban Driving Navigation [[TNNLS2021]](https://ieeexplore.ieee.org/document/9537641)
    
- End-to-End Model-Free Reinforcement Learning for Urban Driving Using Implicit Affordances [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Toromanoff_End-to-End_Model-Free_Reinforcement_Learning_for_Urban_Driving_Using_Implicit_Affordances_CVPR_2020_paper.html)
    
- Toward Deep Reinforcement Learning without a Simulator: An Autonomous Steering Example [[AAAI2018]](https://ojs.aaai.org/index.php/AAAI/article/view/11490)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
## Policy Distillation

<!-- <details><summary>(Click for details)</summary> -->

- DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2308.00398)[[Code]](https://github.com/OpenDriveLab/DriveAdapter)![](https://img.shields.io/github/stars/OpenDriveLab/DriveAdapter?style=social&label=Star)

- Coaching a Teachable Student [[CVPR2023]](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Coaching_a_Teachable_Student_CVPR_2023_paper.html)

- Detrive: Imitation Learning with Transformer Detection for End-to-End Autonomous Driving [[DISA2023]](https://arxiv.org/abs/2310.14224)

- Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline [[NeurIPS2022]](https://arxiv.org/abs/2206.08129)[[code]](https://github.com/OpenDriveLab/TCP)![](https://img.shields.io/github/stars/OpenDriveLab/TCP.svg?style=social&label=Star&maxAge=2592000)

- Learning from All Vehicles [[CVPR2022]](http://arxiv.org/pdf/1709.04622v4)[[Code]](https://github.com/dotchen/LAV.git)![](https://img.shields.io/github/stars/dotchen/LAV.svg?style=social&label=Star&maxAge=2592000)

- End-to-End Urban Driving by Imitating a Reinforcement Learning Coach [[ICCV2021]](https://arxiv.org/abs/2108.08265)[[Code]](https://github.com/zhejz/carla-roach.git)![](https://img.shields.io/github/stars/zhejz/carla-roach.svg?style=social&label=Star&maxAge=2592000)
    
- Learning To Drive From a World on Rails [[ICCV2021]](http://arxiv.org/pdf/2105.00636v3)[[Code]](https://github.com/dotchen/WorldOnRails.git)![](https://img.shields.io/github/stars/dotchen/WorldOnRails.svg?style=social&label=Star&maxAge=2592000)

- Learning by Cheating [[CoRL2020]](http://arxiv.org/pdf/2107.00123v1)[[Code]](https://github.com/dotchen/LearningByCheating.git)![](https://img.shields.io/github/stars/dotchen/LearningByCheating.svg?style=social&label=Star&maxAge=2592000)

- SAM: Squeeze-and-Mimic Networks for Conditional Visual Driving Policy Learning [[CoRL2020]](https://arxiv.org/abs/1912.02973)[[Code]](https://github.com/twsq/sam-driving.git)![](https://img.shields.io/github/stars/twsq/sam-driving.svg?style=social&label=Star&maxAge=2592000)

- Learning to Steer by Mimicking Features from Heterogeneous Auxiliary Networks [[AAAI2019]](http://arxiv.org/pdf/1811.02759v1)[[Code]](https://github.com/cardwing/Codes-for-Steering-Control.git)![](https://img.shields.io/github/stars/cardwing/Codes-for-Steering-Control.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
## Causal Confusion

<!-- <details><summary>(Click for details)</summary> -->

- DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2308.00398)[[Code]](https://github.com/OpenDriveLab/DriveAdapter)![](https://img.shields.io/github/stars/OpenDriveLab/DriveAdapter?style=social&label=Star)

- Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes [[arxiv2023]](https://arxiv.org/abs/2305.10430)

- Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming [[ICML2022]](https://arxiv.org/abs/2206.10816)

- Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction [[ECCV2022]](https://link.springer.com/chapter/10.1007/978-3-031-19842-7_23)

- Object-Aware Regularization for Addressing Causal Confusion in Imitation Learning [[NeurIPS2021]](https://arxiv.org/abs/2110.14118)[[Code]](https://github.com/alinlab/oreo.git)![](https://img.shields.io/github/stars/alinlab/oreo.svg?style=social&label=Star&maxAge=2592000)

- Keyframe-Focused Visual Imitation Learning [[ICML2021]](https://arxiv.org/abs/2106.06452)[[Code]](https://github.com/AlvinWen428/keyframe-focused-imitation-learning)![](https://img.shields.io/github/stars/AlvinWen428/keyframe-focused-imitation-learning.svg?style=social&label=Star&maxAge=2592000)

- Fighting Copycat Agents in Behavioral Cloning from Observation Histories [[NeurIPS2020]](http://arxiv.org/pdf/2010.14876v1)

- Shortcut Learning in Deep Neural Networks [[NatureMachineIntelligence2020]](https://www.nature.com/articles/s42256-020-00257-z)

- Causal Confusion in Imitation Learning [[NeurIPS2019]](https://proceedings.neurips.cc/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html)

- ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst [[RSS2019]](https://arxiv.org/abs/1812.03079)

- Exploring the Limitations of Behavior Cloning for Autonomous Driving [[ICCV2019]](https://arxiv.org/abs/1904.08980)[[Code]](https://github.com/felipecode/coiltraine.git)![](https://img.shields.io/github/stars/felipecode/coiltraine.svg?style=social&label=Star&maxAge=2592000)

- Off-Road Obstacle Avoidance through End-to-End Learning [[NeurIPS2005]](https://proceedings.neurips.cc/paper/2005/hash/fdf1bc5669e8ff5ba45d02fded729feb-Abstract.html)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Robustness

### Long-tailed Distribution

<!-- <details><summary>(Click for details)</summary> -->

- CAT: Closed-loop Adversarial Training for Safe End-to-End Driving [[CoRL2023]](https://openreview.net/forum?id=VtJqMs9ig20)

- Adversarial Driving: Attacking End-to-End Autonomous Driving [[IV2023]](https://arxiv.org/abs/2103.09151)[[Code]](https://github.com/wuhanstudio/adversarial-driving.git)![](https://img.shields.io/github/stars/wuhanstudio/adversarial-driving.svg?style=social&label=Star&maxAge=2592000)

- KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients [[ECCV2022]](https://arxiv.org/abs/2204.13683)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.html)

- TrafficSim: Learning To Simulate Realistic Multi-Agent Behaviors [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Suo_TrafficSim_Learning_To_Simulate_Realistic_Multi-Agent_Behaviors_CVPR_2021_paper.html)

- Multimodal Safety-Critical Scenarios Generation for Decision-Making Algorithms Evaluation [[RAL2021]](https://arxiv.org/abs/2009.08311)

- Learning by Cheating [[CoRL2020]](http://arxiv.org/pdf/2107.00123v1)[[Code]](https://github.com/dotchen/LearningByCheating.git)![](https://img.shields.io/github/stars/dotchen/LearningByCheating.svg?style=social&label=Star&maxAge=2592000)

- Learning to Collide: An Adaptive Safety-Critical Scenarios Generating Method [[IROS2020]](https://arxiv.org/abs/2003.01197)

- Enhanced Transfer Learning for Autonomous Driving with Systematic Accident Simulation [[IROS2020]](https://arxiv.org/abs/2007.12148)

- Improving the Generalization of End-to-End Driving through Procedural Generation [[arXiv2020]](https://arxiv.org/abs/2012.13681)[[Code]](https://github.com/decisionforce/pgdrive.git)![](https://img.shields.io/github/stars/decisionforce/pgdrive.svg?style=social&label=Star&maxAge=2592000)

- Generating Adversarial Driving Scenarios in High-Fidelity Simulators [[ICRA2019]](https://ieeexplore.ieee.org/abstract/document/8793740)

- Scalable End-to-End Autonomous Vehicle Testing via Rare-event Simulation [[NeurIPS2018]](https://proceedings.neurips.cc/paper/2018/hash/653c579e3f9ba5c03f2f2f8cf4512b39-Abstract.html)

- Microscopic Traffic Simulation using SUMO [[ITSC2018]](https://ieeexplore.ieee.org/abstract/document/8569938)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
### Covariate Shift

<!-- <details><summary>(Click for details)</summary> -->

- Exploring Data Aggregation in Policy Learning for Vision-Based Urban Autonomous Driving [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.html)

- Learning by Cheating [[CoRL2020]](http://arxiv.org/pdf/2107.00123v1)[[Code]](https://github.com/dotchen/LearningByCheating.git)![](https://img.shields.io/github/stars/dotchen/LearningByCheating.svg?style=social&label=Star&maxAge=2592000)

- Agile Autonomous Driving using End-to-End Deep Imitation Learning [[RSS2018]](https://arxiv.org/abs/1709.07174)

- Query-Efficient Imitation Learning for End-to-End Simulated Driving [[AAAI2017]](https://ojs.aaai.org/index.php/AAAI/article/view/10857)

- Meta learning Framework for Automated Driving [[arXiv2017]](http://arxiv.org/pdf/1706.04038v1)

- A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning [[AISTATS2011]](http://proceedings.mlr.press/v15/ross11a)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

### Domain Adaptation

<!-- <details><summary>(Click for details)</summary> -->
- DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2308.00398)[[code]](https://github.com/OpenDriveLab/DriveAdapter)![](https://img.shields.io/github/stars/OpenDriveLab/DriveAdapter?style=social&label=Star)

- Learning to Drive Anywhere [[CoRL2023]](https://arxiv.org/abs/2309.12295)

- SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation [[CVPR2022]](https://arxiv.org/abs/2206.08367)[[code]](https://github.com/SysCV/shift-dev)

- Learning Interactive Driving Policies via Data-driven Simulation [[ICRA2022]](https://ieeexplore.ieee.org/abstract/document/9812407)

- Segmented Encoding for Sim2Real of RL-based End-to-End Autonomous Driving [[IV2022]](https://ieeexplore.ieee.org/abstract/document/9827374)

- Domain Adaptation In Reinforcement Learning Via Latent Unified State Representation [[AAAI2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17251)[[Code]](https://github.com/KarlXing/LUSR.git)![](https://img.shields.io/github/stars/KarlXing/LUSR.svg?style=social&label=Star&maxAge=2592000)

- A Versatile and Efficient Reinforcement Learning Framework for Autonomous Driving [[arxiv2021]](https://arxiv.org/abs/2110.11573)

- Enhanced Transfer Learning for Autonomous Driving with Systematic Accident Simulation [[IROS2020]](https://arxiv.org/abs/2007.12148)

- Simulation-Based Reinforcement Learning for Real-World Autonomous Driving [[ICRA2020]](https://ieeexplore.ieee.org/abstract/document/9196730)[[Code]](https://github.com/deepsense-ai/carla-birdeye-view.git)![](https://img.shields.io/github/stars/deepsense-ai/carla-birdeye-view.svg?style=social&label=Star&maxAge=2592000)
    
- Learning to Drive from Simulation without Real World Labels [[ICRA2019]](https://arxiv.org/abs/1812.03823)
    
- Visual-based Autonomous Driving Deployment from a Stochastic and Uncertainty-aware Perspective [[IROS2019]](https://ieeexplore.ieee.org/abstract/document/8968307)

- Virtual to Real Reinforcement Learning for Autonomous Driving [[BMVC2017]](https://arxiv.org/abs/1704.03952)[[Code]](https://github.com/SullyChen/Autopilot-TensorFlow.git)![](https://img.shields.io/github/stars/SullyChen/Autopilot-TensorFlow.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Affordance Learning

<!-- <details><summary>(Click for details)</summary> -->

- Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer [[CoRL2022]](https://arxiv.org/abs/2207.14024)[[Code]](https://github.com/opendilab/InterFuser)![](https://img.shields.io/github/stars/opendilab/InterFuser.svg?style=social&label=Star&maxAge=2592000)

- Multi-Task Learning With Attention for End-to-End Autonomous Driving [[CVPRWorkshop2021]](https://arxiv.org/abs/2104.10753)

- Driver Behavioral Cloning for Route Following in Autonomous Vehicles Using Task Knowledge Distillation [[TIV2022]](https://ieeexplore.ieee.org/abstract/document/9857598)

- Policy-Based Reinforcement Learning for Training Autonomous Driving Agents in Urban Areas With Affordance Learning [[TITS2021]](https://ieeexplore.ieee.org/abstract/document/9599578)

- Conditional Affordance Learning for Driving in Urban Environments [[CoRL2018]](https://proceedings.mlr.press/v87/sauer18a.html)[[code]](https://github.com/xl-sr/CAL)![](https://img.shields.io/github/stars/xl-sr/CAL.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## BEV

<!-- <details><summary>(Click for details)</summary> -->

- BEVGPT: Generative Pre-trained Large Model for Autonomous Driving Prediction, Decision-Making, and Planning [[AAAI2024]](https://arxiv.org/abs/2310.10357)

- DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2308.00398)[[code]](https://github.com/OpenDriveLab/DriveAdapter)![](https://img.shields.io/github/stars/OpenDriveLab/DriveAdapter?style=social&label=Star)

- Planning-oriented Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2212.10156)[[Code]](https://github.com/OpenDriveLab/UniAD)![](https://img.shields.io/github/stars/OpenDriveLab/UniAD.svg?style=social&label=Star&maxAge=2592000)
  
- Think Twice before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2305.06242)[[code]](https://github.com/OpenDriveLab/ThinkTwice)![](https://img.shields.io/github/stars/OpenDriveLab/ThinkTwice.svg?style=social&label=Star&maxAge=2592000)

- Coaching a Teachable Student [[CVPR2023]](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Coaching_a_Teachable_Student_CVPR_2023_paper.html)

- ReasonNet: End-to-End Driving with Temporal and Global Reasoning [[CVPR2023]](https://arxiv.org/abs/2305.10507)

- VAD: Vectorized Scene Representation for Efficient Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2303.12077)[[code]](https://github.com/hustvl/VAD)![](https://img.shields.io/github/stars/hustvl/VAD.svg?style=social&label=Star&maxAge=2592000)

- FusionAD: Multi-modality Fusion for Prediction and Planning Tasks of Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2308.01006)

- UAP-BEV: Uncertainty Aware Planning using Bird's Eye View generated from Surround Monocular Images [[CASE2023]](https://arxiv.org/abs/2306.04939)[[code]](https://github.com/Vikr-182/UAP-BEV)![](https://img.shields.io/github/stars/Vikr-182/UAP-BEV.svg?style=social&label=Star&maxAge=2592000)

- Enhance Sample Efficiency and Robustness of End-to-end Urban Autonomous Driving via Semantic Masked World Model [[NeurIPSWorkshop2022]](https://arxiv.org/abs/2210.04017)

- Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer [[CoRL2022]](https://arxiv.org/abs/2207.14024)[[Code]](https://github.com/opendilab/InterFuser)![](https://img.shields.io/github/stars/opendilab/InterFuser.svg?style=social&label=Star&maxAge=2592000)

- Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning [[ICML2022]](https://arxiv.org/abs/2207.10295)
    
- Learning Mixture of Domain-Specific Experts via Disentangled Factors for Autonomous Driving Authors [[AAAI2022]](https://ojs.aaai.org/index.php/AAAI/article/view/20000)

- ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning [[ECCV2022]](https://arxiv.org/abs/2207.07601)[[Code]](https://github.com/OpenDriveLab/ST-P3)![](https://img.shields.io/github/stars/OpenDriveLab/ST-P3.svg?style=social&label=Star&maxAge=2592000)

- TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving [[TPAMI2022]](https://arxiv.org/abs/2205.15997)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)
        
- Learning from All Vehicles [[CVPR2022]](https://arxiv.org/abs/2203.11934)[[Code]](https://github.com/dotchen/LAV.git)![](https://img.shields.io/github/stars/dotchen/LAV.svg?style=social&label=Star&maxAge=2592000)

- Deep Federated Learning for Autonomous Driving [[IV2022]](http://arxiv.org/pdf/2110.05754v2)[[Code]](https://github.com/aioz-ai/FADNet.git)![](https://img.shields.io/github/stars/aioz-ai/FADNet.svg?style=social&label=Star&maxAge=2592000)
   
- NEAT: Neural Attention Fields for End-to-End Autonomous Driving [[ICCV2021]](https://arxiv.org/abs/2109.04456)[[Code]](https://github.com/autonomousvision/neat.git)![](https://img.shields.io/github/stars/autonomousvision/neat.svg?style=social&label=Star&maxAge=2592000)
    
- ObserveNet Control: A Vision-Dynamics Learning Approach to Predictive Control in Autonomous Vehicles [[RAL2021]](https://arxiv.org/abs/2107.08690)
    
- Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D [[ECCV2020]](https://arxiv.org/abs/2008.05711)[[Code]](https://github.com/nv-tlabs/lift-splat-shoot.git)![](https://img.shields.io/github/stars/nv-tlabs/lift-splat-shoot.svg?style=social&label=Star&maxAge=2592000)

- Driving Through Ghosts: Behavioral Cloning with False Positives [[IROS2020]](https://arxiv.org/abs/2008.12969)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Transformer

<!-- <details><summary>(Click for details)</summary> -->

- Planning-oriented Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2212.10156)[[Code]](https://github.com/OpenDriveLab/UniAD)![](https://img.shields.io/github/stars/OpenDriveLab/UniAD.svg?style=social&label=Star&maxAge=2592000)
  
- Think Twice before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving [[CVPR2023]](https://arxiv.org/abs/2305.06242)[[code]](https://github.com/OpenDriveLab/ThinkTwice)![](https://img.shields.io/github/stars/OpenDriveLab/ThinkTwice.svg?style=social&label=Star&maxAge=2592000)

- ReasonNet: End-to-End Driving with Temporal and Global Reasoning [[CVPR2023]](https://arxiv.org/abs/2305.10507)

- Hidden Biases of End-to-End Driving Models [[ICCV2023]](https://arxiv.org/abs/2306.07957)[[code]](https://github.com/autonomousvision/carla_garage)![](https://img.shields.io/github/stars/autonomousvision/carla_garage.svg?style=social&label=Star&maxAge=2592000)

- VAD: Vectorized Scene Representation for Efficient Autonomous Driving [[ICCV2023]](https://arxiv.org/abs/2303.12077)[[code]](https://github.com/hustvl/VAD)![](https://img.shields.io/github/stars/hustvl/VAD.svg?style=social&label=Star&maxAge=2592000)

- Detrive: Imitation Learning with Transformer Detection for End-to-End Autonomous Driving [[DISA2023]](https://arxiv.org/abs/2310.14224)

- Ground then Navigate: Language-guided Navigation in Dynamic Scenes [[arXiv2022]](https://arxiv.org/abs/2209.11972)

- Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer [[CoRL2022]](https://arxiv.org/abs/2207.14024)[[Code]](https://github.com/opendilab/InterFuser)![](https://img.shields.io/github/stars/opendilab/InterFuser.svg?style=social&label=Star&maxAge=2592000)

- MMFN: Multi-Modal-Fusion-Net for End-to-End Driving [[IROS2022]](https://arxiv.org/abs/2207.00186)[[Code]](https://github.com/Kin-Zhang/mmfn.git)![](https://img.shields.io/github/stars/Kin-Zhang/mmfn.svg?style=social&label=Star&maxAge=2592000)

- TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving [[TPAMI2022]](https://arxiv.org/abs/2205.15997)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- Human-AI Shared Control via Policy Dissection [[NeurIPS2022]](https://arxiv.org/abs/2206.00152)[[Code]](https://github.com/Mehooz/vision4leg.git)

- COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles [[CVPR2022]](https://arxiv.org/abs/2205.02222)[[Code]](https://github.com/UT-Austin-RPL/Coopernaut.git)![](https://img.shields.io/github/stars/UT-Austin-RPL/Coopernaut.svg?style=social&label=Star&maxAge=2592000)

- CADRE: A Cascade Deep Reinforcement Learning Framework for Vision-Based Autonomous Urban Driving [[AAAI2022]](https://arxiv.org/abs/2202.08557)[[Code]](https://github.com/BIT-MCS/Cadre.git)![](https://img.shields.io/github/stars/BIT-MCS/Cadre.svg?style=social&label=Star&maxAge=2592000)

- Safe Driving via Expert Guided Policy Optimization [[CoRL2022]](http://arxiv.org/pdf/2110.06831v2)[[Code]](https://github.com/decisionforce/EGPO.git)![](https://img.shields.io/github/stars/decisionforce/EGPO.svg?style=social&label=Star&maxAge=2592000)

- NEAT: Neural Attention Fields for End-to-End Autonomous Driving [[ICCV2021]](https://arxiv.org/abs/2109.04456)[[Code]](https://github.com/autonomousvision/neat.git)![](https://img.shields.io/github/stars/autonomousvision/neat.svg?style=social&label=Star&maxAge=2592000)

- Multi-Modal Fusion Transformer for End-to-End Autonomous Driving [[CVPR2021]](https://arxiv.org/abs/2104.09224)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>
  
## V2V Cooperative

<!-- <details><summary>(Click for details)</summary> -->

- CADRE: A Cascade Deep Reinforcement Learning Framework for Vision-Based Autonomous Urban Driving [[AAAI2022]](https://arxiv.org/abs/2202.08557)[[Code]](https://github.com/BIT-MCS/Cadre.git)![](https://img.shields.io/github/stars/BIT-MCS/Cadre.svg?style=social&label=Star&maxAge=2592000)

- COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles [[CVPR2022]](https://arxiv.org/abs/2205.02222)[[Code]](https://github.com/UT-Austin-RPL/Coopernaut.git)![](https://img.shields.io/github/stars/UT-Austin-RPL/Coopernaut.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Distributed RL

<!-- <details><summary>(Click for details)</summary> -->

- Safe Driving via Expert Guided Policy Optimization [[CoRL2022]](http://arxiv.org/pdf/2110.06831v2)[[Code]](https://github.com/decisionforce/EGPO.git)![](https://img.shields.io/github/stars/decisionforce/EGPO.svg?style=social&label=Star&maxAge=2592000)

- GRI: General Reinforced Imitation and its Application to Vision-Based Autonomous Driving [[arXiv2021]](https://arxiv.org/abs/2111.08575)
    
- End-to-End Model-Free Reinforcement Learning for Urban Driving Using Implicit Affordances [[CVPR2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Toromanoff_End-to-End_Model-Free_Reinforcement_Learning_for_Urban_Driving_Using_Implicit_Affordances_CVPR_2020_paper.html)
    
- Batch Policy Learning under Constraints [[ICML2019]](http://arxiv.org/pdf/1903.08738v1)[[Code]](https://github.com/gwthomas/force.git)![](https://img.shields.io/github/stars/gwthomas/force.svg?style=social&label=Star&maxAge=2592000)

<!-- </details> -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Data-driven Simulation

### Parameter Initialization

- KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients [[ECCV2022]](https://arxiv.org/abs/2204.13683)[[Code]](https://github.com/autonomousvision/transfuser.git)![](https://img.shields.io/github/stars/autonomousvision/transfuser.svg?style=social&label=Star&maxAge=2592000)

- TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios [[ICRA2023]](https://arxiv.org/abs/2210.06609)[[code]](https://github.com/metadriverse/trafficgen)![](https://img.shields.io/github/stars/metadriverse/trafficgen.svg?style=social&label=Star&maxAge=2592000)

- AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.html)

- SceneGen: Learning To Generate Realistic Traffic Scenes [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Tan_SceneGen_Learning_To_Generate_Realistic_Traffic_Scenes_CVPR_2021_paper.html)

- HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Mi_HDMapGen_A_Hierarchical_Graph_Generative_Model_of_High_Definition_Maps_CVPR_2021_paper.html)

- SimNet: Learning Reactive Self-driving Simulations from Real-world Observations [[ICRA2021]](https://arxiv.org/abs/2105.12332)

- Learning to Collide: An Adaptive Safety-Critical Scenarios Generating Method [[IROS2020]](https://arxiv.org/abs/2003.01197)

<p align="right">(<a href="#top">back to top</a>)</p>

### Traffic Simulation

- Data-driven Traffic Simulation: A Comprehensive Review [[arXiv2023]](https://arxiv.org/abs/2310.15975)

- Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion [[NeurIPS2023]](https://arxiv.org/abs/2311.02738)

- MixSim: A Hierarchical Framework for Mixed Reality Traffic Simulation [[CVPR2023]](https://openaccess.thecvf.com/content/CVPR2023/html/Suo_MixSim_A_Hierarchical_Framework_for_Mixed_Reality_Traffic_Simulation_CVPR_2023_paper.html)

- Learning Realistic Traffic Agents in Closed-loop [[CoRL2023]](https://arxiv.org/abs/2311.01394)

- TrafficBots: Towards World Models for Autonomous Driving Simulation and Motion Prediction [[arXiv2023]](https://arxiv.org/abs/2303.04116)

- Language Conditioned Traffic Generation [[arXiv2023]](https://arxiv.org/abs/2307.07947)[[code]](https://github.com/Ariostgx/lctgen)![](https://img.shields.io/github/stars/Ariostgx/lctgen.svg?style=social&label=Star&maxAge=2592000)

- TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios [[ICRA2023]](https://arxiv.org/abs/2210.06609)[[code]](https://github.com/metadriverse/trafficgen)![](https://img.shields.io/github/stars/metadriverse/trafficgen.svg?style=social&label=Star&maxAge=2592000)

- DriveSceneGen: Generating Diverse and Realistic Driving Scenarios from Scratch [[arXiv2023]](https://arxiv.org/abs/2309.14685)

- Guided Conditional Diffusion for Controllable Traffic Simulation [[arXiv2022]](https://arxiv.org/abs/2210.17366)

- BITS: Bi-level Imitation for Traffic Simulation [[arXiv2022]](https://arxiv.org/abs/2208.12403)

- TrafficSim: Learning To Simulate Realistic Multi-Agent Behaviors [[CVPR2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Suo_TrafficSim_Learning_To_Simulate_Realistic_Multi-Agent_Behaviors_CVPR_2021_paper.html)

- SimNet: Learning Reactive Self-driving Simulations from Real-world Observations [[ICRA2021]](https://arxiv.org/abs/2105.12332)

<p align="right">(<a href="#top">back to top</a>)</p>

### Sensor Simulation

- UniSim: A Neural Closed-Loop Sensor Simulator [[CVPR2023]](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.html)

- Learning Compact Representations for LiDAR Completion and Generation [[CVPR2023]](https://openaccess.thecvf.com/content/CVPR2023/html/Xiong_Learning_Compact_Representations_for_LiDAR_Completion_and_Generation_CVPR_2023_paper.html)

- Adv3D: Generating Safety-Critical 3D Objects through Closed-Loop Simulation [[CoRL2023]](https://arxiv.org/abs/2311.01446)

- Reconstructing Objects in-the-wild for Realistic Sensor Simulation [[ICRA2023]](https://arxiv.org/abs/2311.05602)

- Enhancing Photorealism Enhancement [[TPAMI2023]](https://arxiv.org/abs/2105.04619)[[code]](https://github.com/isl-org/PhotorealismEnhancement)![](https://img.shields.io/github/stars/isl-org/PhotorealismEnhancement.svg?style=social&label=Star&maxAge=2592000)

- UrbanGIRAFFE: Representing Urban Scenes as Compositional Generative Neural Feature Fields [[ICCV2023]](https://arxiv.org/abs/2303.14167)[[code]](https://github.com/freemty/urbanGIRAFFE)![](https://img.shields.io/github/stars/freemty/urbanGIRAFFE.svg?style=social&label=Star&maxAge=2592000)

- NeuRAD: Neural Rendering for Autonomous Driving [[arXiv2023]](https://arxiv.org/abs/2311.15260)[[code]](https://github.com/georghess/NeuRAD)![](https://img.shields.io/github/stars/georghess/NeuRAD.svg?style=social&label=Star&maxAge=2592000)

- MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving [[CICAI2023]](https://arxiv.org/abs/2307.15058)[[code]](https://github.com/OPEN-AIR-SUN/mars)![](https://img.shields.io/github/stars/OPEN-AIR-SUN/mars.svg?style=social&label=Star&maxAge=2592000)

- Mega-NERF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/html/Turki_Mega-NERF_Scalable_Construction_of_Large-Scale_NeRFs_for_Virtual_Fly-Throughs_CVPR_2022_paper.html)

- Panoptic Neural Fields: A Semantic Object-Aware Neural Scene Representation [[CVPR2022]](https://openaccess.thecvf.com/content/CVPR2022/html/Kundu_Panoptic_Neural_Fields_A_Semantic_Object-Aware_Neural_Scene_Representation_CVPR_2022_paper.html)

- CADSim: Robust and Scalable in-the-wild 3D Reconstruction for Controllable Sensor Simulation [[CoRL2022]](https://openreview.net/forum?id=Mp3Y5jd7rnW)

- VISTA 2.0: An Open, Data-driven Simulator for Multimodal Sensing and Policy Learning for Autonomous Vehicles [[ICRA2022]](https://arxiv.org/abs/2111.12083)[[code]](https://github.com/vista-simulator/vista)![](https://img.shields.io/github/stars/vista-simulator/vista.svg?style=social&label=Star&maxAge=2592000)

- Learning Interactive Driving Policies via Data-driven Simulation [[ICRA2022]](https://arxiv.org/abs/2111.12137)[[code]](https://github.com/vista-simulator/vista)![](https://img.shields.io/github/stars/vista-simulator/vista.svg?style=social&label=Star&maxAge=2592000)


- Learning Robust Control Policies for End-to-End Autonomous Driving From Data-Driven Simulation [[RAL2020]](https://ieeexplore.ieee.org/abstract/document/8957584)

<p align="right">(<a href="#top">back to top</a>)</p>
