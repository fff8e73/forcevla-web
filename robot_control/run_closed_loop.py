"""CLI and web child-process entry for bounded stop-observe-replan sessions."""

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Keep CPU preprocessing from introducing oversubscribed BLAS pools in children.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from robot_control.closed_loop import ClosedLoop, atomic_json
from robot_control.closed_loop_contracts import Cancellation, LoopConfig
from robot_control.closed_loop_native import (
    NativeRosMotionTransport,
    NativeSimulatedIO,
    create_session,
)
from robot_control.closed_loop_observation import FileObservationSource
from robot_control.closed_loop_simulation import (
    SimulatedObservationSource,
    SyntheticGeometry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["simulate", "live"], required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/closed_loop_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Insert the Ethernet cable into the Ethernet port on the end face of the docking station.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--parent-control", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.config.read_text())
    if args.cycles is not None:
        data["max_cycles"] = args.cycles
    config = LoopConfig.from_dict(data)
    runtime = json.loads(args.runtime.read_text()) if args.runtime else {}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "status.json").exists() or (
        args.output / "events.jsonl"
    ).exists():
        parser.error("Output already contains a run; old runs cannot resume")
    os.chmod(args.output, 0o700)
    sys.setswitchinterval(0.0002)
    cancel = Cancellation()
    seen = [time.monotonic()]

    def signal_stop(signum, frame):
        cancel.cancel("service_stopping")

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    if args.parent_control:

        def reader():
            try:
                while True:
                    line = sys.stdin.readline(2048)
                    if not line:
                        cancel.cancel("parent_disconnected")
                        return
                    command = json.loads(line)
                    if command.get("command") == "stop":
                        cancel.cancel("user_cancelled")
                    elif command.get("command") == "heartbeat":
                        value = command.get("seen_monotonic")
                        if (
                            type(value) not in (int, float)
                            or not 0
                            <= time.monotonic() - value
                            <= config.browser_disconnect_timeout_s
                        ):
                            cancel.cancel("browser_disconnected")
                        else:
                            seen[0] = max(seen[0], value)
            except Exception:
                cancel.cancel("parent_disconnected")

        threading.Thread(target=reader, name="parent-control", daemon=True).start()
    session = io = None
    try:
        if args.mode == "simulate":
            io = NativeSimulatedIO(
                config, args.output / "native", SyntheticGeometry().bounds
            )
            source = SimulatedObservationSource(
                io, config, freeze_after=runtime.get("simulation_freeze_after")
            )
            bounds = SyntheticGeometry().bounds
        else:
            from robot_control.native_qp import NativeQP

            source = FileObservationSource(runtime["ipc_path"], config)
            limits = json.loads(Path(runtime["planning_config"]).read_text())
            bounds = NativeQP(limits["expected_qp_parameters"], limits).bounds
            io = NativeRosMotionTransport(
                config,
                args.output / "native",
                bounds,
                runtime.get("qualification_path"),
            )
            deadline = time.monotonic() + config.observation_timeout_s
            while True:
                try:
                    io.feedback()
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
        session = create_session(io, bounds, config, args.output / "execution")
        loop = ClosedLoop(
            source,
            session,
            config,
            args.output,
            mode=args.mode,
            runtime=runtime,
            prompt=args.prompt,
            seed=args.seed,
            cancellation=cancel,
            client_seen=(lambda: seen[0]) if args.parent_control else None,
        )
        result = loop.run()
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("id", "state", "completed_cycles", "reason", "synthetic")
                },
                ensure_ascii=False,
            )
        )
        return 0 if result["state"] == "COMPLETED_LIMIT" else 1
    except Exception as exc:
        outcome, stop = "FAILED", None
        if session:
            try:
                stop = session.stop("startup_or_finalization_failure")
                if not stop.get("confirmed"):
                    outcome = "STOP_UNCONFIRMED"
                elif session.status()["phase"] != "RELEASED":
                    session.release()
            except Exception as stopped:
                outcome, stop = (
                    "STOP_UNCONFIRMED",
                    dict(confirmed=False, error=str(stopped)),
                )
            try:
                session.close()
            except Exception:
                pass
        elif io:
            io.close()
        atomic_json(
            args.output / "status.json",
            dict(
                id=args.output.name,
                state=outcome,
                stop=stop,
                mode=args.mode,
                synthetic=args.mode == "simulate",
                reason=str(exc),
                completed_cycles=0,
                config=data,
                finished_unix=time.time(),
                hardware_commands_sent=0 if session is None else None,
            ),
        )
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
