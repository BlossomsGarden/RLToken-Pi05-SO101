# RLToken-Pi05-SO101

SO-101 双臂、双摄颜色方块抓取与叠放的 Step 1–5 工作仓库。SFT 使用四种颜色组合共 120 集；RL Token **只使用红块叠绿块的一种组合、30 集**（episode 0–29），任务指令为 “Pick up the red block and place it on top of the green block.”。权重、演示视频、标注 JSON 与标定文件留在原设备，不进入 Git。

**结果边界：** SFT 对红→黄、绿→黄有过初步可行观察，没有增加新颜色，但尚无严格成功率统计；这不能证明对任意未见组合泛化。远程私有 20k checkpoint 和真实机械臂未能在本仓库离线复验。下文的可复现入口不等于真实服务器运行记录。

## 文件与环境

| 路径 | 用途 |
| --- | --- |
| pi05-SFT-操作手册.md、sft/ | Step 1–2 的原始手册、SO-101 SFT 配方、固定噪声 OpenPI 服务、Windows 推理客户端、转换脚本 |
| scripts/、configs/ | RL-Token-Pi05-open 的 Step 3–5 训练和部署脚本，以及本仓库新增的双机通信入口与 30 集清单 |
| lerobot/ | **完整**修改版 LeRobot 0.6.2 源码，直接复制进仓库，不使用第三方包封装或 Git 子模块 |
| tests/、RLT-操作手册.md | 本地契约测试、双机部署与按键工作记录 |

服务器使用两个**独立虚拟环境**：JAX SFT 环境安装 OpenPI commit 15a9616，用于 Step 2/2.5 与 6006 服务；另一个 Python >=3.12 的 PyTorch/CUDA 环境安装本仓库的 lerobot/，用于 Step 3–5 与 6016 服务，例如在仓库根目录运行 python -m pip install -e ./lerobot。Windows 主机使用与 LeLab 标定匹配的环境，按需安装 requirements-client.txt；无需安装 GPU 权重。requirements-ssh.txt 是补充依赖，不包含完整 JAX、CUDA Torch 或本地 LeRobot。

迁入的 LeRobot 功能包括：lerobot/src/lerobot/policies/pi05_rlt/ 中的 RL Token 编码器、解码器、Actor、双 Critic、关键阶段检测及 VLA 嵌入适配；lerobot/src/lerobot/policies/pi05/ 的状态/图像、相对动作、q01/q99 归一化和 RTC 接口；OpenCV 摄像头的 FOURCC/预热以及 SO-101 主臂位置反馈。scripts/rlt_core.py 和 train_rlt_stage*.py 提供回放、训练与 checkpoint 结构；本仓库另加分离式服务。以上描述是**迁入代码包含的功能**，不是未经核验的上游逐行 diff。原始许可证及归属见 NOTICE、LICENSE 和 lerobot/ 文件头。

## Step 1 — 标定与数据

使用 LeLab portable win64 标定 SO-101，记录双摄 30 fps 抓取和叠放演示。数据集 BlossomsGarden/pick-and-place-v3_20261025_051506 含四种颜色组合各 30 集。部署前核对相机顺序、主从臂 COM 号及标定 ID；详情见 pi05-SFT-操作手册.md。数据集不复制进仓库。

## Step 2 — JAX 全量 SFT

远程实际训练代码和权重未提供；sft/so101_config.py 与 so101_policy.py 是根据用户手册和 So101-Pi0.5-Order-Generalization-main 整理的**复现配方**，并非远程脚本逐字备份。配置：4×H200 数据并行，每卡 batch 4、global 16、workers 8；AdamW、clip 1.0、峰值 lr 2.5e-5、warmup 1000、末期 lr 2.5e-6、全参数包括 PaliGemma 图像/语言模块、全参数 EMA 0.999，训练到 20k steps。

前五关节相对模型生成 action chunk 时的 observation.state 输出 delta，夹爪输出绝对值；使用 q01/q99。外置相机映射 base_0_rgb，第二相机映射 left_wrist_0_rgb，第三路全零并关闭 mask；640×480 缩至 224×168，再上下各 pad 28 行至 224×224，动作与相机对齐为 30 fps。

服务器 OpenPI 环境内，在仓库根目录运行（按实际服务器路径替换）：

    export OPENPI_ROOT=/path/to/openpi-15a9616
    export PYTHONPATH="$PWD:$OPENPI_ROOT/src"
    python -m sft.openpi_entry compute_norm_stats --openpi-root "$OPENPI_ROOT" --config-name pi05_so101_pickplace_v3
    python -m sft.openpi_entry train --openpi-root "$OPENPI_ROOT" pi05_so101_pickplace_v3 --exp-name so101_pickplace_v3

20k checkpoint 路径以 /path/to/your/sft/checkpoint 占位。sft/serve_so101.sh 运行仅监听 127.0.0.1:6006 的旧 SFT 服务；sft/test_so101_inference.py 用合成观测检查输出形状；sft/deploy_openpi.py 是旧 Windows 推理入口。通过 SSH 本地端口转发连接，实机参数仍以 SFT 手册为准。

## Step 2.5 — JAX 转 PyTorch

sft/convert_jax_model_to_pytorch.py 与 OpenPI commit 15a9616 自带 examples/convert_jax_model_to_pytorch.py 一致。服务器 JAX 环境可通过注册本地 SO-101 配方的入口运行：

    python -m sft.openpi_entry convert --openpi-root "$OPENPI_ROOT" --checkpoint_dir /path/to/your/sft/checkpoint --config_name pi05_so101_pickplace_v3 --output_path /path/to/your/pi05_converted

注意官方转换脚本用 checkpoint_dir 路径字符串中是否包含 pi05 判断部分模型结构；上述占位路径**不是**已验证的真实转换命令。切换到独立 PyTorch 环境后，sft/prepare_rl_checkpoint.py 可以为转换产物建立独立 LeRobot 配置和权重软链，不覆盖原文件：

    python -m sft.prepare_rl_checkpoint --converted /path/to/your/pi05_converted --output /path/to/your/pi05_lerobot_pretrained --norm-stats /path/to/your/sft/assets/BlossomsGarden/pick-and-place-v3_20261025_051506/norm_stats.json

OpenPI 的 action 统计是前五关节 delta 加夹爪绝对值；RL replay/Actor 的 action 统计是原始 LeRobot 数据集的六维绝对动作。服务端需分别输入 --openpi-norm-stats 和 --absolute-stats，不可互换。权重加载和两套归一化在真实 20k checkpoint 上仍待验证。

### 模型格式转换过程实录

待用户补充实际转换命令、检验步骤和遇到的问题；此处不臆造运行记录。

## Step 3 — RL Token

30 集选择见 configs/red_on_green_episodes.json。冻结已转换 π0.5，在服务器 LeRobot 环境预计算 image-only embedding，再训练 Stage 1 RL Token；显式指定相机映射，避免原参考脚本默认 top/wrist 与本数据 front/side 不符：

    python scripts/precompute_pi05_embeddings.py --pi05_path /path/to/your/pi05_lerobot_pretrained --tokenizer_path /path/to/your/tokenizer --dataset_repo_id /path/to/your/lerobot_dataset --output_dir outputs/red_green_embeddings --shard_episodes 0-29 --camera_map '{"observation.images.front":"observation.images.front","observation.images.side":"observation.images.side"}'
    python scripts/train_rlt_stage1_pi05.py --precomputed_path outputs/red_green_embeddings --output_dir outputs/rlt_stage1 --device cuda

训练产物留在服务器。scripts/verify_rlt_checkpoints.py 可在有真实权重时做结构兼容性检查。

## Step 4 — 标注与关键阶段

用户已在本机用原仓库 scripts/annotate_dataset.py 为 120 集制作标注，输出 annotations_pick_and_place_120.json 并上传 GPU 服务器。此文件未提供给本仓库，不应推断标注区间。迁移后示意命令（保留当时数据与 Python 路径）：

    $repo = 'D:\2026offer\so101\RLToken-Pi05-SO101'
    $data = 'D:\2026offer\so101\sources\lelab-main\.cache\huggingface\lerobot\BlossomsGarden\pick-and-place-v3_20261025_051506'
    $py = 'D:\2026offer\so101\sources\lelab-main\.venv\Scripts\python.exe'
    & $py "$repo\scripts\annotate_dataset.py" --dataset_path $data --annotations_out "$repo\annotations_pick_and_place_120.json" --camera observation.images.front --force_sheet --sheet_dir "$repo\annotation_sheets"

在服务器只挑出红→绿集后，用 precompute_critical_zrl.py、train_critical_classifier.py 和 extract_annotated_warmup.py 完成 Stage 4 和 warmup journal：

    python scripts/select_red_green_annotations.py --annotations /path/to/your/annotations_pick_and_place_120.json --out /path/to/your/annotations_red_green_30.json
    python scripts/precompute_critical_zrl.py --pi05_path /path/to/your/pi05_lerobot_pretrained --tokenizer_path /path/to/your/tokenizer --rlt_checkpoint /path/to/your/rlt_stage1/best_checkpoint.pt --dataset_path /path/to/your/lerobot_dataset --annotations /path/to/your/annotations_red_green_30.json --output outputs/critical_zrl.npz
    python scripts/train_critical_classifier.py --precomputed_zrl outputs/critical_zrl.npz --output outputs/critical_classifier.pt
    python scripts/extract_annotated_warmup.py --dataset_path /path/to/your/lerobot_dataset --annotations /path/to/your/annotations_red_green_30.json --pi05_path /path/to/your/pi05_lerobot_pretrained --tokenizer_path /path/to/your/tokenizer --rlt_checkpoint /path/to/your/rlt_stage1/best_checkpoint.pt --output outputs/red_green_warmup.pkl

## Step 5 — 分离式在线 RL 与推理

原版 train_rlt_stage2_pi05.py、infer_rlt_stage2_pi05.py 和 eval_rlt_pi05_so101.py 留作单机参考。新增 scripts/distributed_gpu.py 在服务器计算 π0.5、关键阶段及 Actor 动作，scripts/distributed_learner.py 异步更新 Actor/Critic；scripts/distributed_windows.py 在本机采集双摄和关节、执行 10 步窗口、驱动主臂镜像、处理 HIL 与奖励。只在动作窗口边界进行网络请求，客户端每步目标 30 fps。**该分离实现尚未使用私有权重和真机端到端验证**，不可与原版单机实测混为一谈。

服务器 LeRobot 环境运行训练服务（--absolute-stats 是原始数据集 meta/stats.json）：

    python -m scripts.distributed_gpu --mode train --pi05-path /path/to/your/pi05_lerobot_pretrained --rlt-checkpoint /path/to/your/rlt_stage1/best_checkpoint.pt --openpi-norm-stats /path/to/your/sft/norm_stats.json --absolute-stats /path/to/your/lerobot_dataset/meta/stats.json --critical-classifier /path/to/your/critical_classifier.pt --restore-replay /path/to/your/red_green_warmup.pkl --output outputs/distributed_stage2 --port 6016

服务只绑定 127.0.0.1:6016；本地另开并保持 SSH 隧道，密钥与服务器参数仅在本机提供：

    ssh -N -L 6016:127.0.0.1:6016 -p <SSH端口> <SSH用户>@<服务器IP> -i <密钥路径>

核对 configs/distributed_windows.json 的主从 COM 号、摄像头索引 1/0、标定 ID，先做连接 dry run，再由操作者明确启用 --execute：

    python -m scripts.distributed_windows --config configs/distributed_windows.json --port 6016
    python -m scripts.distributed_windows --config configs/distributed_windows.json --port 6016 --episodes 5 --execute

h 切换人工接管，主臂自主时跟随从臂；c 切换关键阶段，a 允许 Actor（仍需服务器达到更新阈值），f 完成本集，g 放弃本集，e 停机并释放力矩，q 退出。每集结束提示 y/N，人工确认成功时 terminal reward=1；中断集不计入 replay。需先检查两臂对齐、摄像头画面和每关节安全阈值。训练器保存 outputs/distributed_stage2/latest_checkpoint.pt。最终推理用单独的 --mode infer 服务，提供 --actor-checkpoint 并维持相同 SSH 隧道与 Windows 控臂入口，运行前停止旧服务。

## 验证现状

tests/ 包含协议、回放和无硬件检查；CPU 模拟学习器完成过更新和 checkpoint 保存。这些不能代替真实 20k 权重加载、转换、远程联调及真机安全验证。后续应补实际转换过程实录和红→黄、绿→黄有重复次数与明确成功标准的泛化实验。
