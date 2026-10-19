"""Existing inference wire protocol with an explicit observation/request ID."""

import numpy as np


def predict_observation(policy, request, observation_id):
    from examples.forcevla.transport.codec import make_request, decode

    message = make_request(
        request, model_name=policy.model_name, request_id=observation_id
    )
    response = policy.stub.Infer(message, timeout=policy.timeout)
    if response.request_id != observation_id:
        raise RuntimeError("RPC response does not match the observation ID")
    result = {name: decode(value) for name, value in response.outputs.items()}
    for name, shape in [("actions", (1, 50, 14)), ("normalized_actions", (1, 50, 32))]:
        if (
            name not in result
            or result[name].shape != shape
            or not np.isfinite(result[name]).all()
        ):
            raise ValueError("Invalid prediction: " + name)
    result["inference_time_us"] = response.inference_time_us
    return result
