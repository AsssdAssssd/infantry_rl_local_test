"""复用 RSL-RL checkpoint 的 infos 保存各任务成功率窗口与等级。"""

from .config import TASK_NAMES


def _expand_legacy_critic_input(module, state_dict, prefix, *args):
    """迁移旧 critic：新增 one-hot 列置零，保持初始 value 不变。"""
    key = prefix + "mlp.0.weight"
    weight = state_dict.get(key)
    target = module.mlp[0].weight
    if weight is not None and weight.shape == (target.shape[0], target.shape[1] - len(TASK_NAMES)):
        expanded = weight.new_zeros(target.shape)
        expanded[:, :weight.shape[1]] = weight
        state_dict[key] = expanded
        print("[train] expanded legacy critic input with zero-initialized task one-hot weights")


def task_runner_class(base, workers=None):
    class TaskRunner(base):
        def _configure_multi_gpu(self):
            if workers is None or not workers.distributed:
                return super()._configure_multi_gpu()
            # Startup already initialized NCCL to coordinate resume and run paths.
            # Unlike the upstream runner, each worker sees only its assigned GPU.
            if self.device != "cuda:0":
                raise ValueError(f"Runner device {self.device} does not match the isolated GPU cuda:0")
            self.is_distributed = True
            self.gpu_global_rank = workers.rank
            self.gpu_local_rank = workers.local_rank
            self.gpu_world_size = workers.world_size
            self.cfg["multi_gpu"] = dict(global_rank=workers.rank, local_rank=workers.local_rank,
                                         world_size=workers.world_size)

        def save(self, path, infos=None):
            infos = dict(infos or {})
            infos["terrain_curriculum"] = self.env.terrain_state_dict()
            return super().save(path, infos)

        def load(self, path, load_cfg=None, strict=True, map_location=None):
            hook = None
            # 仅迁移权重；完整续训必须严格匹配模型和优化器维度。
            if (load_cfg is not None and load_cfg.get("critic", True)
                    and not load_cfg.get("optimizer", True) and not load_cfg.get("iteration", True)
                    and self.env.obs_cfg.get("critic_task_one_hot", False)):
                hook = self.alg._raw_critic.register_load_state_dict_pre_hook(_expand_legacy_critic_input)
            try:
                infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
            finally:
                if hook is not None:
                    hook.remove()
            # --load-weights 明确重启课程；完整恢复时在 set_training_iteration/reset 应用。
            if load_cfg is None or load_cfg.get("iteration", True):
                self.env._pending_terrain_state = (infos or {}).get("terrain_curriculum")
            return infos

    return TaskRunner
