# 在线 WebSocket 协议

Windows 客户端经 SSH 回环端口 6016 向 GPU 服务端发送 MessagePack 二进制帧。连接按请求发出顺序处理，每个请求含单调递增的整数 `req_id`；响应包含相同 `req_id`、`ok` 及数据或 `error`。客户端发送与接收独立，队列与 inflight 上限均为 64；接收按 FIFO 队头匹配 ID，错序或积压立即停机。

## 坐标与动作

- 线上 `state[6]` 和 `actual[N][6]` 均为标定后的**客户端 LeRobot 关节数值**：前五关节为度，夹爪为归一化范围单位；顺序是 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。服务器入口统一执行 `model=scale*client+offset`，模型预测得到的数据集数值在出口执行逆映射；客户端不二次映射。
- 冻结 π0.5 输出 50×6 的**归一化 delta** 参考（前五关节相对该 chunk 起点的观测 state，夹爪绝对）。服务器读取同一个 SFT checkpoint 的 OpenPI `state`／`actions` q01/q99，Actor/critic、replay/BC 使用此归一化空间。执行给机械臂的为逆归一化与锚定后的绝对客户端角度。
- `front`、`side` 是分别对应训练时视角的 JPEG 字节；`critical_steps`、`sources` 对齐已实际下发的每一步动作。只有全窗口标记关键且关闭时关键的 policy 窗口，或全窗口关键的 HUMAN 窗口写入 replay；跨相位窗口丢弃。非关键中间窗口不发 JPEG 边界报文；结束、接管或切换相位时由下一次 plan/snapshot/interrupt 同步跳过的控制步数。

| command | 主要请求字段 | 响应要点 |
| --- | --- | --- |
| `health` | req_id | mode、next_episode、active_episode |
| `plan` | episode、step、actor、manual_critical、state、front/side | plan_id、actions[50][6]、critical、critical_prob、updates |
| `boundary_slice` | episode、plan_id、step、actual、sources、critical_steps、interrupted、state、front/side | 可用的 Actor patch、critical、updates |
| `human_slice` | 同上但所有来源为 HUMAN | critical、updates |
| `snapshot` | h 时的 episode、plan_id、step、state、front/side | 新鲜冻结 VLA 提议只用于后续人工 replay 参考；不返回待执行动作 |
| `interrupt_plan` | episode、plan_id、step | 记录被跳过的非关键步并结束当前父计划 |
| `finish` | episode、step、reward 0/1、discard | next_episode |

每个关键 slice 最多 10 步、严格按顺序写入。`h` 的本地接管不等待服务器；旧关键段先发，之后发 `snapshot`，后续 HUMAN slice 依 FIFO 处理。人工阶段分类器不更新滞回状态；`s` 等人工报文确认后用当前观测重新 `plan`。Actor 需通过在线集数、更新数、关键相位持续步数三重门槛；仅以 Actor mean 作为执行目标，在关键阶段的每个 10 步边界阻塞等待修正，其他时候执行完整 50 步 VLA chunk。

`finish` 的 reward 仅加入最后一条有效 replay transition 的最终有效步；`discard` 不进入训练、不消耗 episode 号，也不写本地档。客户端保存 `rlt_episodes/ep_XXXX.npz`，服务器保存 `replay_journal.pkl` 与 checkpoint；异常恢复从 checkpoint 的 journal 长度截断尚未提交的尾部。若结束时离最后一个关键 replay 窗口已有非关键动作，该奖励的归属需在生产机复核（见 `需要进一步确定.md`）。
