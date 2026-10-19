"""Step-four simulation and read-only readiness audit; live hardware stays disabled."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from robot_control.common import ROOT, load_config, new_run, save_json, source_manifest
from robot_control.command_bridge import CommandBridge, Prediction, current_action, ControlFault
from robot_control.hardware_backend import SimulatedBackend, readiness_audit


def simulation(limits, fault=None):
    # Synthetic fixture, never altered real/model actions. Idealized software test.
    initial = np.array(
        [
            0.5,
            0.1,
            0.7,
            0,
            0,
            0,
            0.5,
            0,
            0,
            0,
            0,
            0,
            0,
            0.5,
            -0.1,
            0.7,
            0,
            0,
            0,
            0.5,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=float,
    )
    backend = SimulatedBackend(initial)
    bridge = CommandBridge(backend, limits)
    now = 10.0
    period = 1 / limits["control_hz"]
    try:
        bridge.connect(now)
        bridge.request("synthetic-request", now, now)
        # Explicit virtual-time event loop: inference completion is independent of hold ticks.
        observed = now
        completion = now + 0.25
        for i in range(200):
            now = 10.0 + i * period
            if i == 50:
                if fault == "rpc_timeout":
                    completion = float("inf")
                elif fault:
                    backend.fault = fault
            if now >= completion and bridge.state == "WAITING":
                actions = np.tile(current_action(initial), (50, 1))
                actions[:, 0] += np.arange(50) * 0.0001
                bridge.submit(
                    Prediction("synthetic-request", observed, actions, True), now
                )
            bridge.tick(now)
            if bridge.state in ["STOPPED", "FAULT_UNCONFIRMED"]:
                break
            if any(e["event"] == "chunk_complete" for e in bridge.events):
                break
        if fault == "rpc_timeout":
            for i in range(200, 650):
                now = 10.0 + i * period
                bridge.tick(now)
                if bridge.state in ["STOPPED", "FAULT_UNCONFIRMED"]:
                    break
    except (ControlFault, ConnectionError) as exc:
        bridge.event(now, "simulation_fault_caught", error=str(exc))
    finally:
        bridge.close(now)
    return {
        "scope": "synthetic_idealized_controller_virtual_time_not_robot_or_QP_simulation",
        "status": "simulation_completed",
        "state": bridge.state,
        "fault_reason": bridge.fault_reason,
        "configured_fault": fault,
        "submitted_ticks": bridge.sent,
        "stop_receipts": bridge.stop_receipts,
        "events": bridge.events,
        "backend_events": backend.events,
        "hardware_commands_sent": 0,
        "limits": limits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["audit", "simulate", "hardware"], default="audit"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.json")
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=ROOT / "config/control_simulation.json",
    )
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fault",
        choices=[
            "owner_lost",
            "gripper_owner_lost",
            "disconnect",
            "stale_feedback",
            "tracking_error",
            "ik_failure",
            "right_write_failure",
            "gripper_reject",
            "stop_unconfirmed",
            "rpc_timeout",
        ],
    )
    args = parser.parse_args()
    output = new_run(args.output, "control-" + args.mode)
    report = {
        "status": "failed",
        "started_unix_ns": time.time_ns(),
        "source_sha256": source_manifest(),
        "hardware_commands_sent": 0,
    }
    code = 1
    try:
        if args.mode == "simulate":
            if args.prediction_dir:
                raise ValueError(
                    "Simulation uses an explicitly synthetic fixture, not replayed real predictions"
                )
            limits = json.loads(args.simulation_config.read_text())
            report.update(simulation(limits, args.fault))
            code = 0
        else:
            if args.fault:
                raise ValueError("Fault injection is simulation-only")
            config = load_config(args.config)
            save_json(output / "config.json", config)
            report.update(readiness_audit(config, args.prediction_dir))
            report["requested_mode"] = args.mode
            # No option, file flag, or accepted prediction can instantiate a live writer.
            if args.mode == "hardware":
                report["execution_error"] = (
                    "Live backend unavailable; no publishers created, no ownership/mode changes"
                )
            code = 2
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        report["finished_unix_ns"] = time.time_ns()
        save_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(output / "report.json"),
                "hardware_commands_sent": 0,
                "blocking_reasons": report.get("blocking_reasons"),
                "error": report.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
