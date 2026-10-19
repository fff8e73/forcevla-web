"""Remote-only model adapter. Does not load CUDA or model weights."""

import sys
import time

import numpy as np


class RemotePolicy:
    def __init__(self, config):
        sys.path.insert(0, config["policy_repo"])
        from examples.forcevla.deployment.policy import ForceVLARemotePolicy
        from examples.forcevla.transport import codec

        self.client = ForceVLARemotePolicy(config["model_address"], timeout=config["rpc_timeout_s"])
        self.health_client = ForceVLARemotePolicy(config["model_address"], timeout=1.0)
        self.codec = codec

    def healthy(self):
        return self.health_client.healthy()

    def predict(self, raw, request_id):
        request = self.codec.make_request(
            raw, model_name=self.client.model_name, request_id=request_id
        )
        response = self.client.stub.Infer(request, timeout=self.client.timeout)
        if response.request_id != request_id:
            raise ValueError("RPC request ID mismatch")
        result = {name: self.codec.decode(value) for name, value in response.outputs.items()}
        for key, shape in (("actions", (1, 50, 14)), ("normalized_actions", (1, 50, 32))):
            if (
                key not in result
                or result[key].shape != shape
                or not np.isfinite(result[key]).all()
            ):
                raise ValueError(f"Invalid {key} response")
        result["inference_time_us"] = response.inference_time_us
        return result

    def close(self):
        self.client.close()
        self.health_client.close()


class TianjiPi05Policy:
    """Remote PhyAI policy for the reviewed Tianji right-arm bundle.

    The web console only owns the client socket. Model process lifecycle is
    managed by ``ModelManager`` so a model can be selected without importing
    CUDA or ROS into the web process.
    """

    def __init__(self, config, endpoint):
        repo = config["phyai_repository"]
        sys.path.insert(0, repo)
        from phyai_robot.remote import RemotePolicy as PhyAIRemotePolicy

        self.client = PhyAIRemotePolicy(
            endpoint,
            timeout_s=config.get("rpc_timeout_s", 10.0),
            warmup_timeout_s=config.get("warmup_timeout_s", 120.0),
        )
        self.spec = None

    def info(self):
        self.spec = self.client.info()
        return self.spec

    def healthy(self):
        try:
            self.info()
            return True
        except Exception:
            return False

    def predict(self, raw, request_id):
        if self.spec is None:
            self.info()
        inputs = {
            "image.head_left": raw["image"],
            "image.right_wrist": raw["right_wrist_image"],
            "state": np.asarray(raw["state"][13:21], dtype=np.float32),
        }
        self.client.begin_run()
        try:
            self.client.warmup(inputs)
            result = self.client.predict(inputs)
            actions = np.asarray(result["actions"], dtype=np.float32)
        finally:
            self.client.close()
        if actions.shape != (50, 8) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid Tianji Pi0.5 action shape: {actions.shape}")
        full = np.zeros((50, 14), dtype=np.float32)
        full[:, :7] = np.asarray(raw["state"][:7], dtype=np.float32)
        full[:, 7] = np.asarray(raw["state"][6], dtype=np.float32)
        full[:, 7:14] = actions[:, :7]
        full[:, 13] = actions[:, 7]
        return {
            "actions": full[None],
            "normalized_actions": np.zeros((1, 50, 32), dtype=np.float32),
            "inference_time_us": 0,
        }

    def close(self):
        self.client.close()


class DemoPolicy:
    def healthy(self):
        return True

    def predict(self, raw, request_id):
        time.sleep(0.35)
        action = np.tile(np.r_[raw["state"][:7], raw["state"][13:20]], (50, 1))
        action[:, 0] += np.linspace(0, 0.08, 50)
        action[:, 7] += np.linspace(0, 0.05, 50)
        action[:, 2] += np.sin(np.linspace(0, np.pi, 50)) * 0.025
        return {
            "actions": action[None].astype(np.float32),
            "normalized_actions": np.zeros((1, 50, 32), np.float32),
            "inference_time_us": 350000,
        }

    def close(self):
        pass
