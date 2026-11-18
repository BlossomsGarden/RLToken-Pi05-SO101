# 🤖 RLToken 在线训练操作手册

任务固定为**红块叠绿块并对齐**：在 Windows 用双相机、SO-101 从臂及主臂介入，服务器用冻结的 20k SFT π0.5、RLToken 表示与 Stage 2 Actor/Critic 在线训练。全流程见 [总 README](README.md)，前置模型见 [SFT 操作手册](SFT-操作手册.md)，动作及网络协议见 [PROTOCOL](PROTOCOL.md)。

## 🗺️ 双机训练数据流

```text
Windows 真机（30 fps）                               Linux GPU 服务器（单卡 H200）
┌─────────────────────────────────────┐             ┌─────────────────────────────────────┐
│ front / side 双摄 + 从臂状态 6 维    │  JPEG/状态   │ 冻结 SFT π0.5 → 50 步参考动作        │
│ 主臂软镜像；h 即时接管 / s 恢复       │────────────→│ 冻结 Stage 3 编码器 → z_rl           │
│ c 关键相位；f→y/n 终局稀疏奖励        │ SSH -L 6016 │ Stage 4 分类器或手动相位             │
│ 从臂安全层 + 10 步边界数据记录        │←────────────│ 按 a + 门槛：Actor mean 修正关键窗    │
│ 本地 rlt_episodes/ep_XXXX.npz       │ 目标/相位    │ 关键窗口 → replay → 异步 learner     │
└─────────────────────────────────────┘             └─────────────────────────────────────┘
```

- 🧊 冻结 SFT π0.5、Stage 3 编码器和 Stage 4 分类器；只训练 Actor 与双 Critic。优化目标为**双 Q + 目标动作噪声 + 10×BC−Q**，没有标准 SAC 熵项。
- 🔁 `--no-replan-every-window`：10 步边界异步提交 replay，不逐窗等待；**仅执行 VLA 和 HUMAN，不执行 Actor**。50 步连续执行完再停稳、重规划；队列上限 64。要测试 Actor，去掉此参数，以 10 步阻塞重规划并手动按 `a`；在逐窗模式中迟到的 patch 仅能覆盖尚未执行的本窗口动作。
- 🎮 `h` 即时切到主臂，立刻取冻结 VLA 参考快照；HUMAN 阶段每 10 步刷新一张供下一窗使用。`s` 等待已发窗口确认，再用当前观测重规划。
- 📌 replay 取**关闭边界**关键标记；混合相位窗按末端标记处理。50 步末窗延至下一 plan 确认相位再转换；`f` 后停稳、拍最终快照并补报末窗，奖励落在最后一条有效 transition 的最终执行步。非关键尾段不补报。

## ⚗️ 环境与资产

- 🖥️ 服务器：沿用原训练服务器，在独立的 **Python 3.12 + CUDA PyTorch** 环境安装本仓库的 `lerobot/` 和 [requirements-ssh.txt](requirements-ssh.txt)；使用单卡 H200，端口 6016 只监听 `127.0.0.1`。不要在 JAX SFT 环境里混装 PyTorch 栈。
- 🦾 Windows：沿用 LeLab GUI 的 `.venv` 并安装 [requirements-client.txt](requirements-client.txt)；顶层 `deploy_rlt.py`、`check_camera.py` 与 `windows/`、`rlt_online/`、`sft/`、`configs/` 保持同一仓库相对结构。
- 📦 离线输入：SFT 20k checkpoint、匹配的 OpenPI q01/q99 stats、红块叠绿块的前 **30** 集、单跨度关键区间 JSON。`configs/rlt_online.json` 默认引用 `checkpoints/pi05_lerobot_pretrained`、`checkpoints/pi05_norm_stats.json`、`outputs/rlt_stage1/best_checkpoint.pt`、`outputs/critical_classifier.pt`；这些权重/数据**不随代码包提供**。
- 🛡️ `configs/joint_mapping.json` 已按原机报告记录六轴比例全 1、偏移 `[1.94, -2.50, -3.33, 8.39, 1.02, -0.53]`；仅供**与当前设备逐轴复核**。`configs/robot.json` 六轴物理限位仍为 `null`，三项核验标志均为 `false`；实测安全验收前 dry-run 和 `--execute` 都被拦截，勿直接改成已验证。

## 🧬 Stage 3：编码器训练

此处的“Stage 3”指**总流程第 3 步**；仓库代码为 `train_rlt_stage1_pi05.py`（RLToken 内部称 Stage 1）。从前 30 集计算 SFT π0.5 的 image-only prefix，冻结 π0.5、训练 RLToken encoder/decoder，保存编码器供在线计算 `z_rl`：

```bash
# 若服务器上只有原 JAX SFT 权重，先从同一个 20k checkpoint 转换
# 在原 OpenPI 环境运行；须确认原 OpenPI checkout 与该 checkpoint 同源
python -m sft.openpi_entry convert --openpi-root ../openpi --checkpoint-dir checkpoints/pi05_20k --config-name pi05_so101_pickplace_v3 --output-path checkpoints/pi05_converted
# 在 PyTorch RL 环境运行：原始 stats 位于 assets/ 下数据集专属的多级子目录
mapfile -t stats_files < <(find checkpoints/pi05_20k/assets -type f -name norm_stats.json)
if [ "${#stats_files[@]}" -ne 1 ]; then echo "Expected exactly one checkpoint norm_stats.json" >&2; exit 1; fi
sha256sum "${stats_files[0]}"
cp -- "${stats_files[0]}" checkpoints/pi05_norm_stats.json
cmp "${stats_files[0]}" checkpoints/pi05_norm_stats.json
python -m sft.prepare_rl_checkpoint --converted checkpoints/pi05_converted --output checkpoints/pi05_lerobot_pretrained --norm-stats checkpoints/pi05_norm_stats.json

python scripts/precompute_pi05_embeddings.py --pi05_path checkpoints/pi05_lerobot_pretrained --tokenizer_path checkpoints/tokenizer --dataset_repo_id datasets/red_green --output_dir outputs/red_green_embeddings --shard_episodes 0-29 --camera_map '{"observation.images.front":"observation.images.front","observation.images.side":"observation.images.side"}'
python scripts/train_rlt_stage1_pi05.py --precomputed_path outputs/red_green_embeddings --output_dir outputs/rlt_stage1 --device cuda
```

`--norm-stats` 与在线配置 `openpi_norm_stats` 必须指向同源原始统计；此命令保留单一数据来源，并用 `cmp` 核实复制一致。原机报告的同源 20k 文件 SHA-256 为 `8448c396dbea44418387a7eedff41fbf555521ff02c4d82a36b25a2f642b316d`，仅在使用**该原机同一份文件**时应匹配。前五轴为 delta／夹爪绝对动作的 q01/q99，不可拿 LeRobot 绝对动作重新计算。`--converted` 是同源 JAX 权重转换目录；示例 `checkpoints/`、`datasets/` 均为相对路径。Stage 3 编码器脚本单卡，`--steps` 默认 30000。

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

隧道保持开启；浏览器访问 `http://localhost:9000/windows/` 下载部署文件。Windows 根目录的 `.env` 先按 [SFT 手册示例](SFT-操作手册.md) 设置从臂、相机和 6006，再添加：

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

# 实测并核验主从臂、初始方块位置、物理限位后才允许运动；此模式持续学习但不执行 Actor
python deploy_rlt.py --no-calibration --execute --no-replan-every-window --no-auto-critical

# 要让 Actor 在关键窗口接管：以逐窗重规划模式启动，再在运行中按 a 开启
python deploy_rlt.py --no-calibration --execute --no-auto-critical
```

这条 dry-run **也会检查** Windows 环境、三项 `*_verified` 标志与六轴实测物理限位；先完成真机核验，再填配置，不能仅为绕过检查而把占位值设为已验证。集间先释放主臂**和从臂**力矩供人工摆臂／摆方块；按 Enter 后先回读从臂位置作为目标，再恢复力矩，避免追赶旧目标。每集最长 1500 步（`--steps` 可调）；程序不会自行摆正实体方块。演示纠错时，在进入关键放置区间时按 `c`，需要人类介入立刻按 `h`，完成后按 `s`；显著失败或成功后按 `f` 并标注。自动分类器上线时去掉 `--no-auto-critical`。

| 按键 | 真机作用 |
| --- | --- |
| `Enter` | 每集摆好臂和方块后，开始本集 |
| `a` | 开／关 Actor 执行许可，初始关闭、同一程序跨集保留；仅逐窗重规划模式有效，需同时通过服务端硬门槛 |
| `h` | 即刻接管：从臂跟主臂运动，动作记 `HUMAN`；刷新 VLA 参考快照但不执行它 |
| `s` | 退出接管；确认人工窗口写回后按当前观测重新规划 |
| `c` | 切换关键相位；手动模式保持到再次按键；自动模式 3 秒内不被自动检测覆盖，此后可在下一规划边界按分类器结果更新 |
| `f` | 提前结束本集，随后 `y` 成功／`n` 失败／`Enter` 放弃 |
| `g` | 放弃本集，不进 replay、不占用 episode 号 |
| `q` | 放弃本集并退出；再次启动从本地已有集号续编 |
| `e` | 紧急关从臂力矩并退出；仍需保持现场安全 |

Actor 的**双重门**是操作员按 `a`、服务器达到**5 集带关键 transition 的完成集、replay 至少 256 条才开始 learner 更新、累计更新至少 1000 次、关键期连续至少 20 步**。这只是代码最低门槛；原机实操建议先积累约 **30 集关键 transition** 再按 `a`。默认命令采用平滑模式，只供 VLA + 人工接管的数据采集及后台训练，即使按 `a` 也不会下发 Actor 动作。逐窗模式按 `a` 后若还没过硬门槛也继续执行 VLA；按 `a` 切换后立刻从当前观测重新规划。没有关键窗的 episode 不推进 warmup。看服务端 `metrics.jsonl` 的 `episodes_completed`、`replay`、`updates` 和客户端 `source`；人工救回仍标真实结果，但不计为自主成功。第 0 步按 `f` 无实际动作可标注，会弃集并写异常审计档。

## 📊 记录、恢复与评估

- 💾 Windows 每个非放弃集生成 `rlt_episodes/ep_XXXX.npz`（关节、实际动作、`VLA/ACTOR/HUMAN`、关键标记、终局 reward；不含完整双摄图像）。安全限位、网络等异常时尽力通知 brain 弃集；另存 `rlt_episodes/aborted_ep_XXXX_*.npz`（`aborted=true`、`reward=-1`），只供审计、不计集号、不自动回灌 replay；断连后服务端也清除未提交的活动集。
- 🧠 服务器写 `outputs/rlt_online/replay_journal.pkl`、`metrics.jsonl`、`latest_checkpoint.pt`；每 20 集**有 transition 的完成集**和正常退出保存 checkpoint。`replay_journal.pkl` 与 checkpoint 成对备份。
- 🔄 正常停机：先结束当前集，按 `q` 离开 Windows 客户端，再运行 `bash stop_brain.sh`。继续训练时将 `configs/rlt_online.json` 的 `resume_checkpoint` 指向 `outputs/rlt_online/latest_checkpoint.pt` 并保持 `warmup_journal=null`。异常停机恢复可能截去未提交 journal 尾段；核实本地 `.npz` 和服务器 replay 是否一致。
- 📈 Windows 项目根目录运行 `python -m scripts.summarize_online_metrics --episodes-dir rlt_episodes`；复制服务器 metrics 日志后可加 `--metrics outputs/rlt_online/metrics.jsonl` 汇总更新数和最新 Critic loss。训练稳定性也看关键窗口数、干预比例、连续成功率、控制抖动与相机/网络时延。
- 🧪 独立评估时停训练服务，切到 `mode=infer`、设定 `actor_checkpoint`，用一致初始分布各跑 SFT VLA 与冻结 Actor **无人接管** 的重复试验（建议各 ≥50 集），分别统计对齐成功率、耗时、失败模式；人工救回不计自主成功率。当前没有正式对照结果，所需指标见 [真机验收清单](进一步确认的文档.md)。

用户提供的**原机本地自测**为服务器推理约 231 ms（VLA 184 + `z_rl` 43 + 其他 4）、同机端口往返约 260 ms；均不是 Windows 跨 SSH 隧道实测。本机真机限位、映射精度与无人接管成功率须按 [真机验收清单](进一步确认的文档.md) 核验。
