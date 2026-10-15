# Rotate 二维平移命令

原始 commands 五维：`[speed, base_wz, height, gimbal_yaw_rate, translation_direction]`。
方向单位为弧度，相对云台参考 yaw：0 前方、+pi/2 左方、-pi/2 右方、pi 后方。
目标速度在云台参考系中为 `speed * [cos(direction), sin(direction)]`。
默认课程速度非负，方向均匀采样 [-pi, pi]；若配置负速度则沿指定方向反向运动。
训练所有阶段云台 yaw 速率固定为 0，底盘自旋仍由 base_wz 控制。

actor 保留原四维 commands 观测，另加 `translation_direction=[sin(theta), cos(theta)]`；
原始角度不直接进入网络。静止样本 speed=base_wz=0，方向不影响零目标速度。
速度误差、方向侧向误差和超速惩罚统一沿目标方向计算。

交互评估：上下箭头改速度，左右箭头改底盘角速度，逗号/句号每次调整方向 15 度。
旧四维配置仍使用原来的前后方向语义。新训练观测增加两维，不能直接严格加载旧 actor；
跨接口训练需要单独的权重映射，当前改动不实现 warm_start。

验证覆盖方向采样、课程切换、静止目标、四个基本方向、奖励投影、±pi 连续性、
部分环境重采样及旧四维命令。不代表完成仿真训练验证。
