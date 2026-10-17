"""复用 RSL-RL checkpoint 的 infos 保存各任务成功率窗口与等级。"""


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
            infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
            # --load-weights 明确重启课程；完整恢复时在 set_training_iteration/reset 应用。
            if load_cfg is None or load_cfg.get("iteration", True):
                self.env._pending_terrain_state = (infos or {}).get("terrain_curriculum")
            return infos

    return TaskRunner
