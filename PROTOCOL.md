# 在线 WebSocket 协议

Windows 客户端经 SSH 回环端口 6016 向 GPU 服务端发送 MessagePack 二进制帧。连接按请求发出顺序处理，每个请求含单调递增的整数 `req_id`；响应包含相同 `req_id`、`ok` 及数据或 `error`。客户端发送与接收独立，队列与 inflight 上限均为 64；接收按 FIFO 队头匹配 ID，错序或积压立即停机。

## 坐标与动作

- 线上 `state[6]` 和 `actual[N][6]` 均为标定后的**客户端 LeRobot 关节数值**：前五关节为度，夹爪为归一化范围单位；顺序是 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。服务器入口统一执行 `model=scale*client+offset`，模型预测得到的数据集数值在出口执行逆映射；客户端不二次映射。
- 冻结 π0.5 输出 50×6 的**归一化 delta** 参考（前五关节相对该 chunk 起点的观测 state，夹爪绝对）。服务器读取同一个 SFT checkpoint 的 OpenPI `state`／`actions` q01/q99，Actor/critic、replay/BC 使用此归一化空间。执行给机械臂的为逆归一化与锚定后的绝对客户端角度。
- `front`、`side` 是对应训练视角的 JPEG 字节；`critical_steps`、`sources` 对齐每一步实际下发的动作。关键窗口由**关闭边界是否关键**决定；窗口内标签可以混合。非关键中间窗不发 JPEG；跨过超过 10 步的空档丢弃该窗，下一个有观测的边界重新建立起点。

| command | 主要请求字段 | 响应要点 |
| --- | --- | --- |
| `health` | req_id | mode、next_episode、active_episode |
| `plan` | episode、step、actor、manual_critical、state、front/side | plan_id、actions[50][6]、critical、critical_prob、updates；Actor 须按 a 且启用逐窗重规划 |
| `boundary_slice` | episode、plan_id、step、actual、sources、critical_steps、interrupted；常规窗带 state、front/side；50 步末窗加 defer_close，无图像 | 逐窗模式可返回 Actor patch、critical、updates；末窗先暂存，下一 plan 更新相位后再写入 |
| `human_slice` | 同上但所有来源为 HUMAN | critical、updates |
| `snapshot` | h 时及人工阶段每 10 步的 episode、plan_id、step、state、front/side | 新鲜冻结 VLA 提议只供下一人工窗口参考；不执行 |
| `terminal_snapshot` | 终局 step、plan_id、未上报的实际动作与标签、state、front/side | 停稳后补报末窗，更新最后一条 transition 的 next_ref |
| `interrupt_plan` | episode、plan_id、step | 记录被跳过的非关键步并结束当前父计划 |
| `finish` | episode、step、reward 0/1、discard | next_episode |

每个上报 slice 最多 10 步；客户端连续发送、服务器 FIFO 回包，上限 64。平滑模式（`--no-replan-every-window`）下 10 步边界异步记账，50 步连续执行完才停稳并重新规划，**仅执行 VLA／HUMAN**；按 `a` 仍不会请求 Actor。逐窗重规划模式需要操作员按 `a` 开启 Actor，集间保持该开关，重启程序恢复关闭；服务端还要求 5 集有关键 transition 的完成集、replay ≥256、updates ≥1000、关键期连续 ≥20 步。满足门槛时执行 Actor mean，未满足时仍执行 VLA；实操建议先积累约 30 集关键示教。`h` 不等待服务器，先发送旧窗口，再发送 snapshot；人工阶段分类器不更新滞回，`s` 等待已发报文后重新规划。

`finish` 的 reward 加在最后一条有效 replay transition 的最后有效步；终局前执行的非关键尾段不补报，也不改变奖励位置。人工按 `g/q` 放弃的 `discard` 不计训练、不占 episode 号，也不写普通本地档；中途发生限位／网络异常时，客户端尽力发送 `finish(discard=True)`，并另写 `aborted_ep_*.npz` 审计档（`aborted=true`、`reward=-1`，不回灌 replay）。即使发送失败，连接关闭也会让服务端清掉未提交的活动集。带动作但没有关键 transition 的完成集可留本地档，却不计 warmup。服务端 checkpoint 含 `episodes_completed`，与 journal 成对恢复，崩溃后截掉未提交尾段。
