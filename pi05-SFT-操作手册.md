# **pi05-SFT-操作手册**

```text
┌──────────────────────────────────────────────────────────────┐
│ 本地主机：Windows                                             │
│ So101-Pi0.5-Ball-For-Real + LeRobot 0.6.0                    │
│                                                              │
│ 真机双相机图像 + SO101 从臂 6DoF 关节状态 + 任务指令            │
│                         ↓                                    │
│ 观测采集 → RGB / JPEG 编码 → OpenPI WebSocket 客户端           │
│                                      ↑                       │
│                             接收 H50×6 绝对目标动作           │
│                                      ↓                       │
│                    时效 / 范围 / 关节限速 / 跟踪误差检查        │
│                                      ↓                       │
│                       LeRobot → USB 串口 → SO101 从臂         │
│                                                              │
│ 无硬件测试：合成图像与状态 → 同一客户端 → 模拟执行器             │
└────────────────────┬────────────────────▲────────────────────┘
                     │                    │
                     │ JPEG 双相机         │ H50×6 动作序列
                     │ 6 维状态 / 指令     │ 推理耗时
                     ▼                    │
           SSH 本地端口转发：127.0.0.1:16006 -> 服务器端口 127.0.0.1:6006
                     │                    ▲
                     │                    |
                     │ WebSocket / MessagePack
                     ▼                    │
┌──────────────────────────────────────────────────────────────┐
│ 远程服务器：Linux（4xH200）                                   │
│ conda环境 w JAX + OpenPI commit 15a9616 + OpenPI π0.5 权重    │
| 127.0.0.1:6006 使用定制化的 serve_policy_fixed_noise.py       |
│                                                              │
│ JPEG 解码 → SO101 输入转换 / 归一化 → GPU 模型推理             │
│                                      ↓                       │
│                  反归一化 / 恢复绝对目标 → 返回 H50×6 动作     │
│                                                              │
│ checkpoint: ou05_so101_pickplace_v3 20k step                 │
│ 模型调用链不经过 ROS2 / MuJoCo 容器                            │
└──────────────────────────────────────────────────────────────┘
```

## Preparation

- 🤗 [Portable Lelab GUI](https://huggingface.co/datasets/BlossomsGarden/pick-and-place-v3_20261025_051506/resolve/main/lelab-portable-win64.zip)
- 🤗 Dataset: [BlossomsGarden/pick-and-place-v3\_20261025\_051506](https://huggingface.co/datasets/BlossomsGarden/pick-and-place-v3_20261025_051506)
- ☁️ Base Model (OpenPI-pi0.5):
```powershell
gsutil -m  cp -r   gs://openpi-assets/checkpoints/pi05_base/params   D:\google-cloud-cli\download
```

## ⚗️ 环境配方

- 服务器端训练推理：`requirements-ssh.txt`
- 本机对接控臂：Lelab GUI 现成环境(.venv) + `requirements-client.txt`


## 🧪 SFT 配方

4*H200 DP 每卡batch=4，global=16，num_workers=8，AdamW(clip_norm=1.0)，lr 2.5e-5，warmup 1000至 2.5e-6
全量微调，连Paligamma的llm和img都调了
EMA 0.999 也是全参数
动作空间为前5关节输出delta，夹爪gripper输出绝对值。动作频率与相机对齐，30fps
delta所使用的基准来自模型生成chunk的那个观测的observation.state（施加RTC使用execute-horizon=15,--prefetch-after=5的话那么参考的基准大概就是5~15之间的某帧的状态吧）
q01/q99动作归一化，外置相机->base_0_rgb、腕部相机->left_wrist_0_rgb、全0mask->right_wrist相机
相机图片从640x480等比缩放到168x224，然后上下各pad 28行黑边到224x224，保持30fps

每 10kstep 约4h26min，显存占用约110G



## 👩‍🔬 Inference


pytorch fp32 22G 184ms，
此处使用 jax fp16 8.8G 85ms,
总时延大概120ms左右，相机记录与编码、读关节信息这些 10ms，SSH隧道传输约 20ms，服务器端解码然后推理85ms，再发回控制信息5ms

是否太慢？
不慢吧，主要是看它一次生成50步动作，30fps也就是1.67s，100ms只有10%不到，RTC秒了
不过实际上确实是有一卡一卡的，但是检查的时候说是内部的settle策略，要连续4次检测运动小于0.5°才算停止才会开始抓拍（不然糊掉了）和下一个chunk推理

### 服务器侧：启动推理服务

```bash

bash serve_so101.sh 6006
python test_so101_inference.py --port 6006 --prompt "Pick up the red block and place it on top of the green block."
# 预期：actions shape (50,6), SMOKE TEST PASSED, 稳态 ~100-150ms/请求
```


### Windows侧

SSH 隧道（必须时刻保持开启状态）

```powershell
ssh -p xxx  -L 9000:127.0.0.1:9000 -L 6006:127.0.0.1:6006  root@xx.xx.xx.xx  -i  C:\xx\id_ed25519
```

机械臂标定文件最好保持和数据集收集时的标定文件相同，随后编辑 .env 文件
```
# SO101 真机部署 Windows 环境模板

# ---- 机械臂（设备管理器确认 COM 号） ----
FOLLOWER_PORT=COM5
FOLLOWER_ID=so101
# ps: 此处是 .cache\huggingface\lerobot\calibration\robots\so_follower 底下的 json 文件的文件名

# ---- 双摄（OpenCV 索引；一定确保是否写反ID） ----
FRONT_CAMERA=1
SIDE_CAMERA=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
FPS=30

# ---- lelab 标定目录 ----
HF_LEROBOT_CALIBRATION=D:/2026offer/so101/sources/lelab-main/.cache/huggingface/lerobot/calibration

# ---- 任务指令（后续启动时还可以通过命令行参数覆写） ----
TASK_DESCRIPTION=Pick up the red block and place it on top of the green block.

# ---- 推理服务（SSH 隧道后的本地回环地址） ----
OPENPI_HOST=127.0.0.1
OPENPI_PORT=6006
OPENPI_API_KEY=（空着）

```

确认相机 ID 没有标错
```powershell
python check_cameras.py
```


确认机械臂起始位姿
```powershell
# 数据集 120 episodes 首帧 6 关节均值为 [-3, -103, 93, 55, 0, 0]
python deploy_openpi.py --no-calibration --jpeg-quality 90 --steps 1 --execute-horizon 50 --prefetch-after 50  --fps 30
```


实机执行
```powershell
# 完整 H50
python deploy_openpi.py --no-calibration --jpeg-quality 90 --steps 2000 --execute-horizon 50 --prefetch-after 50  --fps 30 --execute  --task "Pick up the red block and place it on top of the green block."

# RTC 
python deploy_openpi.py --no-calibration --jpeg-quality 90 --steps 2000 --execute-horizon 15 --prefetch-after 5  --fps 30 --execute  --task "Pick up the red block and place it on top of the green block."
```


### Demo & 推理速度

TODO

