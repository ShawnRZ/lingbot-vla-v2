<h1 align="center">LingBot-VLA 2.0：从基础模型到落地应用</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2607.06403"><img src="https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=arxiv"></a>
  <a href="https://technology.robbyant.com/lingbot-vla-v2"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/collections/robbyant/lingbot-vla-v2"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow"></a>
  <a href="https://modelscope.cn/collections/Robbyant/LingBot-VLA-V2"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

## 概述

**LingBot-VLA 2.0** 是一个面向实际应用的视觉-语言-动作（Vision-Language-Action）基础模型，旨在推动模型从大规模预训练走向可靠的真实世界机器人应用。

相比 LingBot-VLA 1.0，LingBot-VLA 2.0 在三个核心能力上做了提升：

- **跨任务、跨本体的泛化能力**：重新设计的数据流水线整理了约 **60,000 小时**预训练数据，其中包括覆盖 **20 种机器人构型**的 **50,000 小时**机器人轨迹数据，以及 **10,000 小时**第一人称人类视频。
- **扩展的动作空间**：统一表征不再局限于标准的双臂操作，而是同时支持手臂、末端执行器、夹爪、灵巧手、腰部、头部以及移动底盘信号。
- **预测性动力学建模**：以未来预测作为代理任务，由 DINO-Video 提供语义时序先验，由 LingBot-Depth 提供几何线索。

<p align="center">
  <img src="assets/lingbot_vla2_framework.png" width="86%">
</p>

## 新闻

- **[2026-07-25]** RoboTwin 后训练权重发布：[lingbot-vla-v2-6b-robotwin](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin)。
- **[2026-07-08]** LingBot-VLA 2.0 技术报告与预训练权重准备就绪。

## 安装

环境要求：

- Miniconda 或 Anaconda
- Python 3.12
- PyTorch 2.8.0

在运行安装脚本前，请确认 Conda 已在你的 shell 中完成初始化，且 `conda activate` 可以正常使用。

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
cd lingbot-vla-v2

bash tools/create_train_env.sh
```

默认情况下，脚本会通过 pip 安装 `flash-attn==2.8.3`。如果你本地已有匹配的 wheel 包，可以显式指定：

```bash
bash tools/create_train_env.sh \
  --flash-attn-wheel /path/to/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

你也可以指定环境名称或强制重建环境：

```bash
bash tools/create_train_env.sh \
  --env-name lingbotvla \
  --recreate
```

## 模型下载

我们以 native-depth 模型的形式发布 **LingBot-VLA 2.0** 预训练权重。

| 模型名称 | Hugging Face | ModelScope | 说明 |
| :--- | :---: | :---: | :---: |
| LingBot-VLA 2.0 | [lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) | [lingbot-vla-v2-6b](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b) | Native Depth |

若要使用本代码库训练 LingBot-VLA 2.0，还需要 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)、[MoGe-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal)、[LingBot-Depth](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/depth) 以及 [DINO-VIDEO](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/dino_video) 的教师模型 checkpoint / 配置文件。详见 [Training_Config.md](configs/vla/Training_Config.md)。

```bash
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir lingbot-vla
```

## 预训练数据

LingBot-VLA 2.0 使用了大规模、异构的预训练语料，覆盖单臂、双臂、半人形、人形以及第一人称视角等多种数据来源。

<p align="center">
  <img src="assets/lingbot_vla2_data_demo.png" width="100%">
</p>

原始数据池经过筛选，形成高质量的机器人数据流与第一人称数据流。机器人侧会剔除视频-状态不对齐、模糊或遮挡的视频、多视角不对齐、速度/加速度/加加速度异常，以及信号静止的 episode。第一人称侧则保留以操作为核心的视频，重建并标准化手部轨迹，并过滤相机或手部运动估计不稳定的片段。

<p align="center">
  <img src="assets/lingbot_vla2_data_process.png" width="70%">
</p>

## 模型设计

### 统一动作表征

LingBot-VLA 2.0 将异构本体映射到一个 55 维的标准状态/动作向量：

- 14 维：手臂关节位置
- 14 维：末端执行器位姿
- 2 维：夹爪位置
- 12 维：手部关节位置
- 4 维：腰部位置
- 2 维：头部位置
- 3 维：移动信号
- 4 维：预留维度

<p align="center">
  <img src="assets/lingbot_vla2_data_dimension.png" width="100%">
</p>

### MoE 动作专家

为提升跨本体的扩展性，LingBot-VLA 2.0 在动作专家中引入了稀疏 MoE 层。通过细粒度的专家切分与共享专家隔离，通用先验与特定本体/任务的专用模式可以在相同的激活计算预算下共存。

<p align="center">
  <img src="assets/lingbot_vla2_loss_mse_comparison.png" width="90%">
</p>

### 双查询蒸馏

LingBot-VLA 2.0 在视觉/文本 token 之后追加了当前与未来两组感知查询（query）。这些查询分别从 LingBot-Depth 和 DINO-Video 蒸馏而来，促使因果推理同时捕捉当前场景的几何结构与未来场景的演化。

<p align="center">
  <img src="assets/lingbot_vla2_vis_distillation.png" width="100%">
</p>

## 后训练示例

### 数据准备

后训练需要三个准备步骤。关于如何定制自己的数据集，完整指南见[自定义数据指南](lingbotvla/data/vla_data/README.md)。

| 步骤 | 说明 | 产物 |
|------|-------------|--------|
| 1. 准备 LeRobot 数据集 | 准备一个 LeRobot v2.1 或 v3.0 数据集目录 | LeRobot 数据集目录 |
| 2. 准备机器人配置 | 定义从原始 states/actions/images 到统一特征空间的特征映射 | `configs/robot_configs/<data_name>.yaml` |
| 3. 计算归一化统计量 | 在你的数据集上计算归一化统计量 | `assets/norm_stats/<name>.json` |

下面以 **RoboTwin 2.0** 的 50 个任务为例，使用 clean 与 randomized 数据一起训练。

- **步骤 1 - RoboTwin 数据**：按照 [RoboTwin2.0 准备说明](experiment/robotwin/README.md)下载并准备数据集。
- **步骤 2 - 机器人配置**：RoboTwin 的特征映射见 [configs/robot_configs/robotwin.yaml](configs/robot_configs/robotwin.yaml)。
- **步骤 3 - 归一化**：预先计算好的统计量位于 `assets/norm_stats/robotwin.json`。若要针对自定义任务子集重新计算，见[自定义数据指南](lingbotvla/data/vla_data/README.md)。

### 训练

我们提供了 **LingBot-VLA 2.0** 在 RoboTwin 2.0 的 50 个任务（clean 与 randomized 数据）上的后训练示例：

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.train_path assets/training_data/robotwin.txt \
  --data.data_name multi \
  --train.output_dir output/
```

该后训练配置针对 MoE 路由，同时使用了序列级辅助损失（`sequence_wise_mode: "per_sequence"`、`sequence_wise_loss_coeff: 1e-3`）与 z-loss（`router_z_loss_coeff: 1e-4`）。这些项可以根据下游任务进行调整或关闭。若要使用 loss-free 的路由方案，请注释掉序列级辅助损失与 z-loss 选项，并设置 `bias_update_speed: 0.00025`。
该后训练配置还启用了 Muon 优化器。Muon 通常能得到收敛更好的 loss，但会增加训练时间。若想改用默认的 AdamW 优化器，注释掉 `optimizer: muon` 即可。

真机场景请参考 native-depth 训练配置 [real_robot.yaml](configs/vla/real_robot/real_robot.yaml)。关于 batch size、梯度累积、checkpointing、深度/视频蒸馏、MoE 以及优化器设置的详细说明，见 [Training_Config.md](configs/vla/Training_Config.md)。

## 评测与部署

### 开环评测

```bash
export QWEN3_PATH=Qwen/Qwen3-VL-4B-Instruct
python scripts/open_loop_eval.py \
  --model_path path_to_posttraining_ckpt \
  --robo_name robotwin \
  --data_path path_to_validation_data \
  --use_length 50
```

开环评测必须指定 `--robo_name`。它用于从 `configs/robot_configs/{robo_name}.yaml` 中选择机器人配置，例如 `--robo_name robotwin` 会使用 `configs/robot_configs/robotwin.yaml`。



### RoboTwin 部署

在把 RoboTwin 仿真依赖与模型推理依赖调通到同一个环境后，我们提供了一条命令即可完成 RoboTwin 2.0 全部 50 个任务评测的脚本：
```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct/ \
EVAL_WORKDIR=/path/to/Robotwin_code/ \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path /path/to/your/post_training_checkpoint \
  --output_base /path/to/your/eval_output \
  --num_per_gpu 2
```
`num_per_gpu` 指定每张 GPU 上可以并发评测的任务数量。请根据可用显存以及机器能承受的通信负载进行调整。

### 真机部署

```bash
export QWEN3VL_PATH=path_to_Qwen3-VL-4B-Instruct
python -m deploy.lingbot_vla_v2_policy \
  --model_path path_to_posttraining_ckpt \
  --use_compile \
  --use_length 25 \
  --port port
```

使用 `deploy.lingbot_vla_v2_policy`，在 NVIDIA GeForce RTX 4090D 上单次推理（**10 步去噪**）耗时约 **130 ms**。

## 性能

### 真机基准测试
LingBot-VLA 2.0 在通用（generalist）设定下，于 GM-100 双臂操作与长程移动操作任务上进行了评测。指标在适用处以「进度分 / 成功率」的形式给出。

#### GM-100 双臂操作

| 平台 | GR00T N1.7 | π<sub>0.5 | LingBot-VLA-1.0 | LingBot-VLA 2.0 |
| :--- | ---: | ---: | ---: | ---: |
| AgileX Cobot Magic | 36.3 / 17.8 | 59.1 / 32.2 | 58.2 / 30.0 | **66.2 / 34.4** |
| Galaxea R1Pro | 16.4 / 5.6 | 27.4 / 8.9 | 32.7 / **15.6** | **34.6 / 15.6** |

<p align="center">
  <img src="assets/lingbot_vla2_gm100_ablation_barplot.png" width="75%">
</p>

#### 长程移动操作

| 本体 | 任务 | 设定 | LingBot-VLA 2.0 | π<sub>0.5 |
| :--- | :--- | :--- | ---: | ---: |
| Astribot S1 | 冰箱整理 | 域内 | **77.1 / 60.0** | 65.3 / 46.7 |
| Astribot S1 | 冰箱整理 | 域外 | **37.0 / 13.3** | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | 灶台清洁 | 域内 | **84.3 / 66.7** | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | 灶台清洁 | 域外 | **67.5 / 40.0** | 62.5 / 33.3 |


### 仿真基准测试
#### RoboTwin 2.0

| 任务 | π<sub>0.5 | LingBot-VLA-1.0  | LingBot-VLA 2.0 |
| :--- | :---: | :---: | :---: | 
| Clean | 82.74%  | 88.56% | **93.52%** |
| Randomized | 76.76%  | 86.68% | **92.80%** |


## 引用

如果本工作对你的研究有帮助，欢迎引用：

```bibtex
@article{lingbotvla2,
      title={From Foundation to Application: Improving VLA Models in Practice}, 
      author={Wei Wu and Fangjing Wang and Fan Lu and He Sun and Shi Liu and Yunnan Wang and Yibin Yan and Yong Wang and Shuailei Ma and Xinyang Wang and Yibin Liu and Shuai Yang and Tianxiang Zhou and Kejia Zhang and Lei Zhou and Cheng Su and Nan Xue and Bin Tan and Han Zhang and Youchao Zhang and Fei Liao and Xing Zhu and Yujun Shen and Kecheng Zheng},
      journal={arXiv preprint arXiv:2607.06403},
      year={2026}
}
```

## 许可证

本项目基于 [Apache-2.0 License](LICENSE) 开源。

## 致谢

我们衷心感谢 [VeOmni](https://arxiv.org/abs/2508.02317) 与 [LeRobot](https://github.com/huggingface/lerobot) 的开发者。本项目受益于他们对开源社区的贡献。
