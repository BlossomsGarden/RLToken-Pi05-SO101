# 🤖 RLToken 在线训练操作手册

任务固定为**红块叠绿块并对齐**：在 Windows 用双相机、SO-101 从臂及主臂介入，服务器用冻结的 20k SFT π0.5、RLToken 表示与 Stage 2 Actor/Critic 在线训练。全流程见 [总 README](README.md)，前置模型见 [SFT 操作手册](pi05-SFT-操作手册.md)，动作及网络协议见 [PROTOCOL](PROTOCOL.md)。

## 🗺️ 双机训练数据流

```text
Windows 真机（30 fps）                               Linux GPU 服务器（单卡 H200）
┌─────────────────────────────────────┐             ┌─────────────────────────────────────┐
│ front / side 双摄 + 从臂状态 6 维    │  JPEG/状态   │ 冻结 SFT π0.5 → 50 步参考动作        │
│ 主臂软镜像；h 即时接管 / s 恢复       │────────────→│ 冻结 Stage 3 编码器 → z_rl           │
│ c 关键相位；f→y/n 终局稀疏奖励        │ SSH -L 6016 │ Stage 4 分类器或手动相位             │
│ 从臂安全层 + 10 步边界数据记录        │←────────────│ 门槛满足：Stage 5 Actor mean 修正    │
│ 本地 rlt_episodes/ep_XXXX.npz       │ 目标/相位    │ 关键窗口 → replay → 异步 learner     │
└─────────────────────────────────────┘             └─────────────────────────────────────┘
```

- 🧊 冻结：SFT π0.5、Stage 3 编码器和 Stage 4 分类器。只训练在线 Actor 与双 Critic；优化代码是**双 Q + 目标动作噪声 + 10×BC−Q**，不含标准 SAC 熵项，详见 [待确认清单](需要进一步确定.md)。
- 🔁 非关键期执行 50 步 VLA 计划；关键期按 10 步窗口传输并等待 Actor 修正。`--no-replan-every-window` 禁止每 10 步重启整段 VLA 规划，避免无谓停顿。
- 🎮 按 `h` 当下即切到主臂；冻结 VLA 额外生成一次**不执行**的参考快照供 HUMAN replay 使用；`s` 用接管结束时的新观测重新规划。
- 📌 只有全窗口处于关键相位的样本写入 replay；`f` 之后 `y` 成功得 1、`n` 失败得 0，奖励加在最后一个有效关键窗口末端。跨相位窗口丢弃；此处尾部奖励归属待生产机复核。

## ⚗️ 环境与资产

- 🖥️ 服务器：沿用原训练服务器，在独立的 **Python 3.12 + CUDA PyTorch** 环境安装本仓库的 `lerobot/` 和 [requirements-ssh.txt](requirements-ssh.txt)；使用单卡 H200，端口 6016 只监听 `127.0.0.1`。不要在 JAX SFT 环境里混装 PyTorch 栈。
- 🦾 Windows：沿用 LeLab GUI 的 `.venv` 并安装 [requirements-client.txt](requirements-client.txt)；顶层 `deploy_rlt.py`、`check_camera.py` 与 `windows/`、`rlt_online/`、`sft/`、`configs/` 保持同一仓库相对结构。
- 📦 离线输入：SFT 20k checkpoint、匹配的 OpenPI q01/q99 stats、红块叠绿块的前 **30** 集、单跨度关键区间 JSON。`configs/rlt_online.json` 默认引用 `checkpoints/pi05_lerobot_pretrained`、`checkpoints/pi05_norm_stats.json`、`outputs/rlt_stage1/best_checkpoint.pt`、`outputs/critical_classifier.pt`；这些权重/数据**不随代码包提供**。
- 🛡️ 替换机械臂前核验 `configs/joint_mapping.json` 的实测映射，以及 `configs/robot.json` 的六轴上下限与 `calibration_verified` / `mapping_verified` / `safety_verified` 三项；未验收不得开启 `--execute`。映射文件当前仍为占位，不要把示例当作标定结果。

## 🧬 Stage 3：编码器训练

此处的“Stage 3”指**总流程第 3 步**；仓库代码为 `train_rlt_stage1_pi05.py`（RLToken 内部称 Stage 1）。从前 30 集计算 SFT π0.5 的 image-only prefix，冻结 π0.5、训练 RLToken encoder/decoder，保存编码器供在线计算 `z_rl`：

```bash
# 若服务器上只有原 JAX SFT 权重，先从同一个 20k checkpoint 转换
# 在原 OpenPI 环境运行；须确认原 OpenPI checkout 与该 checkpoint 同源
python -m sft.openpi_entry convert --openpi-root ../openpi --checkpoint-dir checkpoints/pi05_20k --config-name pi05_so101_pickplace_v3 --output-path checkpoints/pi05_converted
# 在 PyTorch RL 环境运行；norm-stats 必须取该 checkpoint 原始 delta 统计
python -m sft.prepare_rl_checkpoint --converted checkpoints/pi05_converted --output checkpoints/pi05_lerobot_pretrained --norm-stats checkpoints/pi05_norm_stats.json

python scripts/precompute_pi05_embeddings.py --pi05_path checkpoints/pi05_lerobot_pretrained --tokenizer_path checkpoints/tokenizer --dataset_repo_id datasets/red_green --output_dir outputs/red_green_embeddings --shard_episodes 0-29 --camera_map '{"observation.images.front":"observation.images.front","observation.images.side":"observation.images.side"}'
python scripts/train_rlt_stage1_pi05.py --precomputed_path outputs/red_green_embeddings --output_dir outputs/rlt_stage1 --device cuda
```

`--converted` 指同源 JAX 权重转换出的目录；上述 `checkpoints/`、`datasets/` 都是**相对路径约定**，以服务器已有权重和数据实际位置为准。首次运行转换命令前在原服务器确认 OpenPI/Tyro 参数、归一化统计文件与 checkpoint 相符；离线编码器训练仍在 PyTorch 环境。Stage 3 当前训练脚本单卡；参数 `--steps` 默认 30000，具体生产机运行步数与验证结果见 [待确认清单](需要进一步确定.md)。

## 🏷️ Stage 4：关键相位分类器

使用前 30 集红块叠绿块的**对齐前—放下后**单跨度关键区间，JSON 中每条是 `{"episode": 0, "start": 412, "end": 587}`。本机可用 LeLab 的现成 Python 环境启动图像打标；已有标注时直接传输 JSON：

```powershell
powershell.exe -NoProfile -File .\scripts\annotate_red_green.ps1
# 产物：annotations_red_green_alignment_30.json（0–29，逐集一段）
```

服务器离线计算 `z_rl` 并训练分类器，首次上手可使用手动 `c` 切换关键期；`--no-auto-critical` 只是关闭自动切换，现有配置仍需能加载分类器权重：

```bash
python scripts/precompute_critical_zrl.py --pi05_path checkpoints/pi05_lerobot_pretrained --tokenizer_path checkpoints/tokenizer --rlt_checkpoint outputs/rlt_stage1/best_checkpoint.pt --dataset_path datasets/red_green --annotations annotations_red_green_alignment_30.json --output outputs/critical_zrl.npz
python scripts/train_critical_classifier.py --precomputed_zrl outputs/critical_zrl.npz --output outputs/critical_classifier.pt
```

## 🚀 Stage 5：启动在线训练

### 1️⃣ 服务器：启动 6016

确保 `configs/rlt_online.json` 为 `mode=train`，资产路径、固定噪声、映射已核验；默认 `warmup_journal=null`，**不预载 30 集示范 replay**。需要离线预载时必须另行验证统计和 journal 契约。

```bash
# 仓库根目录；GPU 端到端测试需要模型资产，并不控制真机
python -m scripts.test_stage2_e2e
bash start_brain.sh
python -m scripts.test_brain_client --uri ws://127.0.0.1:6016
# 另外查看 outputs/rlt_online/brain.log
```

### 2️⃣ Windows：隧道与相机

在工作区根目录执行预检（进入服务器前的网络规则），SSH 密钥和主机信息仅放个人会话变量。以下命令从仓库根目录运行：

```powershell
Push-Location ../..
powershell.exe -NoProfile -File .\Ensure-IWanRemote.ps1 -Connect
Pop-Location
ssh -p $env:SSH_PORT -L 9000:127.0.0.1:9000 -L 6006:127.0.0.1:6006 -L 6016:127.0.0.1:6016 "root@$env:SSH_HOST" -i $env:SSH_KEY
```

隧道保持开启；浏览器访问 `http://localhost:9000/windows/` 下载部署文件。Windows 根目录的 `.env` 先按 [SFT 手册示例](pi05-SFT-操作手册.md) 设置从臂、相机和 6006，再添加：

```dotenv
LEADER_PORT=COM6
LEADER_ID=so101
RLT_HOST=127.0.0.1
RLT_PORT=6016
```

`LEADER_ID` 必须对应 `calibration/teleoperators/so_leader/` 内实际 JSON 文件的名称（不含 `.json`），`COM6` 仅为参考。确认相机编号、视角和采集时一致：

```powershell
python check_camera.py
# 当前目录会保存 front.png 与 side.png；手动打开图片，核对外置/腕部视角
```

相机字段支持 `FRONT_CAMERA` / `SIDE_CAMERA` 及旧 `CAM_FRONT` / `CAM_SIDE`；如果同一相机的两个名称给出不同编号，程序报错。

### 3️⃣ Dry-run → 正式收集与学习

```powershell
# 连接真实双摄/从臂，检查首步动作但不运动、不消费集号
python deploy_rlt.py --no-calibration

# 准备好主从臂标定、初始方块位置、限位后才允许运动
python deploy_rlt.py --no-calibration --execute --no-replan-every-window --no-auto-critical
```

这条 dry-run **也会检查** Windows 环境、三项 `*_verified` 标志与六轴实测物理限位；先完成真机核验，再填配置，不能仅为绕过检查而把占位值设为已验证。每集开始前**手动**将机械臂和方块放回 demo 起始配置，按 Enter 执行。每集最长 1500 步（`--steps` 可调）；程序不会自行摆正实体方块。演示纠错时，在进入关键放置区间时按 `c`，需要人类介入立刻按 `h`，完成后按 `s`；显著失败或成功后按 `f` 并标注。自动分类器上线时去掉 `--no-auto-critical`。

| 按键 | 真机作用 |
| --- | --- |
| `Enter` | 每集摆好臂和方块后，开始本集 |
| `h` | 即刻接管：从臂跟主臂运动，动作记 `HUMAN`；刷新 VLA 参考快照但不执行它 |
| `s` | 退出接管；确认人工窗口写回后按当前观测重新规划 |
| `c` | 手动切换关键相位；自动模式下是短时手动覆盖 |
| `f` | 提前结束本集，随后 `y` 成功／`n` 失败／`Enter` 放弃 |
| `g` | 放弃本集，不进 replay、不占用 episode 号 |
| `q` | 放弃本集并退出；再次启动从本地已有集号续编 |
| `e` | 紧急关从臂力矩并退出；仍需保持现场安全 |

`a` 不是当前客户端的键位。代码启动时默认允许 Actor，但**完成 5 集、learner 至少更新 1000 次、关键期连续至少 20 步**后才会执行 Actor mean；之前仍是冻结 VLA。第 5 集不是“训练结束”，learner 在满足 warmup 后后台异步更新，检查服务端 `metrics.jsonl` 内 `updates` 和客户端的 `source`。即使接管救回，本集仍必须标注真实最终成功/失败，统计时区分自主成功与 HUMAN 辅助成功。

## 📊 记录、恢复与评估

- 💾 Windows 每个非放弃集生成 `rlt_episodes/ep_XXXX.npz`（关节、实际动作、`VLA/ACTOR/HUMAN`、关键标记、终局 reward；不含完整双摄图像）。
- 🧠 服务器写 `outputs/rlt_online/replay_journal.pkl`、`metrics.jsonl`、`latest_checkpoint.pt`；每 20 集和正常退出保存 checkpoint。`replay_journal.pkl` 与 checkpoint 成对备份，不要只复制其中一份。
- 🔄 正常停机：先结束当前集，按 `q` 离开 Windows 客户端，再运行 `bash stop_brain.sh`。继续训练时将 `configs/rlt_online.json` 的 `resume_checkpoint` 指向 `outputs/rlt_online/latest_checkpoint.pt` 并保持 `warmup_journal=null`。异常停机恢复可能截去未提交 journal 尾段；核实本地 `.npz` 和服务器 replay 是否一致。
- 📈 Windows 项目根目录运行 `python -m scripts.summarize_online_metrics --episodes-dir rlt_episodes`；复制服务器 metrics 日志后可加 `--metrics outputs/rlt_online/metrics.jsonl` 汇总更新数和最新 Critic loss。训练稳定性也看关键窗口数、干预比例、连续成功率、控制抖动与相机/网络时延。
- 🧪 独立评估时停训练服务，切到 `mode=infer`、设定 `actor_checkpoint`，用一致初始分布各跑 SFT VLA 与冻结 Actor **无人接管** 的重复试验（建议各 ≥50 集），分别统计对齐成功率、耗时、失败模式；人工救回不计自主成功率。当前没有生产机评估结果，所需指标见 [需要进一步确定](需要进一步确定.md)。

用户提供的**原机本地自测**为单次服务器推理约 231 ms（VLA 184 + `z_rl` 43 + 其他 4）、同机端口往返约 260 ms，均不是跨 SSH 隧道的 Windows 实测值。
