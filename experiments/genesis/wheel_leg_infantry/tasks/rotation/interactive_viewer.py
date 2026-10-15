"""rotate eval 的键盘控制和 Viewer 实时诊断组件（在 locomotion 基础上加云台）。"""

import math

from genesis.ext.pyrender.constants import FONT_SIZE, TEXT_PADDING, TextAlign
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.vis.viewer_plugins import ViewerPlugin

from ...tools.tof_viewer import tof_overlay_lines


class DiagnosticsOverlay(ViewerPlugin):
    """Draw persistent evaluation diagnostics as separate lines."""

    font_size = 25

    def __init__(self):
        super().__init__()
        self._lines = ()

    def update(self, lines):
        self._lines = tuple(lines)

    def on_draw(self):
        viewer = self.viewer
        lines = self._lines
        if viewer is None or viewer._renderer is None or not lines:
            return

        x = viewer._viewport_size[0] - TEXT_PADDING
        y = viewer._viewport_size[1] - TEXT_PADDING - int(FONT_SIZE * 1.2)
        line_height = int(self.font_size * 1.2)
        for index, line in enumerate(lines):
            viewer._renderer.render_text(
                line,
                x,
                y - index * line_height,
                font_name="SpaceMono-Regular",
                font_pt=self.font_size,
                color=viewer._font_color,
                align=TextAlign.TOP_RIGHT,
            )


class KeyboardCommand:
    """保存键盘指令；含第 4 条云台 yaw 速率命令。"""

    def __init__(self, env):
        self.env = env
        self.lin_vel = 0.0
        self.translation_direction = 0.0
        self.ang_vel = 0.0
        self.height_target = float(env.commands[0, 2].item())
        self.gimbal_yaw_vel = 0.0
        self.lin_step = 0.5
        self.ang_step = 0.5
        self.height_step = 0.05
        self.gimbal_step = 0.5

        viewer = env.scene.viewer
        self.viewer = viewer
        self.diagnostics_overlay = viewer.add_plugin(DiagnosticsOverlay())
        viewer.register_keybinds(
            self._keybind("command_direction_left", Key.COMMA, self._change_direction, math.pi / 12),
            self._keybind("command_direction_right", Key.PERIOD, self._change_direction, -math.pi / 12),
            self._keybind("command_forward", Key.UP, self._change_lin, self.lin_step),
            self._keybind("command_backward", Key.DOWN, self._change_lin, -self.lin_step),
            self._keybind("command_turn_left", Key.LEFT, self._change_ang, self.ang_step),
            self._keybind("command_turn_right", Key.RIGHT, self._change_ang, -self.ang_step),
            self._keybind("command_raise_body", Key.PAGEUP, self._change_height, self.height_step),
            self._keybind("command_lower_body", Key.PAGEDOWN, self._change_height, -self.height_step),
            # 云台速率：用 [ / ] 避开 viewer 默认的 WASD 相机键。
            self._keybind("command_gimbal_left", Key.BRACKETLEFT, self._change_gimbal, self.gimbal_step),
            self._keybind("command_gimbal_right", Key.BRACKETRIGHT, self._change_gimbal, -self.gimbal_step),
            Keybind(
                "command_stop",
                Key.SPACE,
                key_action=KeyAction.PRESS,
                callback=self.stop,
            ),
        )

    @staticmethod
    def _keybind(name, key, callback, amount):
        return Keybind(
            name,
            key,
            key_action=KeyAction.PRESS,
            callback=callback,
            args=(amount,),
        )

    def _change_lin(self, amount):
        lower, upper = self.env.command_cfg["lin_vel_range"]
        self.lin_vel = min(max(self.lin_vel + amount, lower), upper)

    def _change_direction(self, amount):
        if not self.env._has_direction_command():
            return
        angle = math.atan2(math.sin(self.translation_direction + amount),
                           math.cos(self.translation_direction + amount))
        lower, upper = self.env.command_cfg["translation_direction_range"]
        self.translation_direction = min(max(angle, lower), upper)

    def _change_ang(self, amount):
        lower, upper = self.env.command_cfg["ang_vel_range"]
        self.ang_vel = min(max(self.ang_vel + amount, lower), upper)

    def _change_height(self, amount):
        lower, upper = self.env.command_cfg["base_height_range"]
        self.height_target = min(max(self.height_target + amount, lower), upper)

    def _change_gimbal(self, amount):
        lower, upper = self.env.command_cfg["gimbal_yaw_vel_range"]
        # 注意：写回 self.gimbal_yaw_vel，别写错属性名。
        self.gimbal_yaw_vel = min(max(self.gimbal_yaw_vel + amount, lower), upper)

    def stop(self):
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.gimbal_yaw_vel = 0.0

    def write_to_env(self):
        self.env.commands[:, 0] = self.lin_vel
        self.env.commands[:, 1] = self.ang_vel
        self.env.commands[:, 2] = self.height_target
        self.env.commands[:, 3] = self.gimbal_yaw_vel
        if self.env._has_direction_command():
            self.env.commands[:, 4] = self.translation_direction

    def update_caption(self):
        env = self.env
        obs = env.obs_buf[0]
        pd = env.get_pd_diagnostics()
        velocity = env.get_velocity_estimator_diagnostics()
        gimbal = env.get_rotate_diagnostics()
        text = (
            f"command  vx={self.lin_vel:+.2f} m/s  wz={self.ang_vel:+.2f} rad/s  "
            f"direction={math.degrees(self.translation_direction):+.0f} deg  height={self.height_target:.3f} m  gimbal={self.gimbal_yaw_vel:+.2f} rad/s",
            f"state    vx_true={env.base_lin_vel[0, 0].item():+.2f} m/s  "
            f"vx_est={velocity['estimated_forward_velocity'].item():+.2f} m/s  "
            f"vx_world={env.robot.get_vel()[0,0].item():+.2f} m/s  "
            f"wz={env.base_ang_vel[0, 2].item():+.2f} rad/s",
            f"gimbal   joint={gimbal['gimbal_joint'].item():+.3f}  "
            f"world={gimbal['gimbal_world_yaw'].item():+.3f}  "
            f"ref={gimbal['gimbal_yaw_ref'].item():+.3f}  "
            f"err={gimbal['gimbal_yaw_error'].item():+.3f}",
            f"pose     z={env.base_pos[0, 2].item():.3f} m  ground={env.terrain_height[0].item():.3f} m  "
            f"height={env.base_height[0].item():.3f} m",
            f"legs     left={env.leg_length[0, 0].item():.3f} m  right={env.leg_length[0, 1].item():.3f} m",
            f"obs      n={obs.numel()}  min={obs.min().item():+.2f}  max={obs.max().item():+.2f}  "
            f"mean={obs.mean().item():+.2f}",
            *tof_overlay_lines(env),
            f"joint Kd       {format_tensor(pd['joint_kd'], precision=2)}",
            f"joint -Kd*qdot {format_tensor(pd['joint_kd_damping'], precision=2)}",
        )
        self.diagnostics_overlay.update(text)


def format_tensor(tensor, precision=3):
    values = tensor.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{value:+.{precision}f}" for value in values) + "]"
