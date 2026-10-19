"""Server challenges and bounded connection diagnostics; no robot control."""

import hashlib
import json
import logging
import math
import secrets
import time
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from pathlib import Path


class Heartbeats:
    def __init__(
        self,
        data_dir,
        touch,
        *,
        clock=time.monotonic,
        wall=time.time,
        interval_s=1.0,
        response_timeout_s=5.0,
    ):
        self.touch, self.clock, self.wall = touch, clock, wall
        self.interval, self.timeout = interval_s, response_timeout_s
        self.connections = OrderedDict()
        path = Path(data_dir) / "connection-events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("heartbeat." + secrets.token_hex(8))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = RotatingFileHandler(path, maxBytes=4 * 1024 * 1024, backupCount=2)
        self.logger.addHandler(self.handler)

    def event(self, row, event, **fields):
        self.logger.info(
            json.dumps(
                dict(
                    time_unix=self.wall(),
                    monotonic=self.clock(),
                    event=event,
                    connection_id=row["id"],
                    session_hash=row["session_hash"],
                    **fields,
                ),
                allow_nan=False,
            )
        )

    def open(self, owner, *, peer="", user_agent=""):
        # Retain at most 1024 recent connections; never remove an active one.
        for key in list(self.connections):
            if len(self.connections) < 1024:
                break
            if self.connections[key]["state"] == "closed":
                self.connections.pop(key)
        if len(self.connections) >= 1024:
            raise RuntimeError("connection_capacity")
        now = self.clock()
        row = dict(
            id=secrets.token_hex(16),
            owner=owner,
            session_hash=hashlib.sha256(owner.encode()).hexdigest()[:16],
            opened_at=now,
            state="open",
            sequence=0,
            accepted_sequence=0,
            sent_at=None,
            challenge=None,
            pending={},
            last_ack=None,
            last_ack_unix=None,
            max_gap_s=0.0,
            protocol="awaiting_response",
            rtt_ms=None,
            visibility_state="unknown",
            has_focus=None,
            close_code=None,
            close_reason=None,
            client_close=None,
        )
        self.connections[row["id"]] = row
        self.event(row, "connected", peer=str(peer)[:100], user_agent=user_agent[:240])
        return row["id"]

    def challenge(self, connection_id):
        row = self.connections[connection_id]
        now = self.clock()
        if row["sent_at"] is None or now - row["sent_at"] >= self.interval:
            row["pending"] = {
                seq: p for seq, p in row["pending"].items() if now - p["sent_at"] <= self.timeout
            }
            row["sequence"] += 1
            nonce = secrets.token_hex(12)
            row["pending"][row["sequence"]] = dict(nonce=nonce, sent_at=now)
            row["sent_at"] = now
            row["challenge"] = dict(
                protocol="server_challenge_v1",
                connection_id=row["id"],
                sequence=row["sequence"],
                nonce=nonce,
            )
            self.event(
                row,
                "challenge_sent",
                sequence=row["sequence"],
                pending_responses=len(row["pending"]),
            )
        return row["challenge"]

    @staticmethod
    def page_state(value):
        state = value.get("visibility_state")
        result = dict(
            visibility_state=state if state in ("visible", "hidden", "prerender") else "unknown",
            has_focus=value.get("has_focus") if type(value.get("has_focus")) is bool else None,
        )
        elapsed = value.get("since_previous_ack_ms")
        if type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 86400000:
            result["client_ack_gap_ms"] = elapsed
        return result

    def receive(self, connection_id, message):
        row = self.connections[connection_id]
        if row["state"] != "open":
            return False
        if message == "ping":
            # Existing open pages remain compatible until refreshed. Once a page
            # uses challenges, unsolicited legacy pings cannot renew its lease.
            if row["protocol"] == "server_challenge_v1":
                return False
            self._accepted(row, "legacy_client_ping")
            return True
        if len(message) > 4096:
            return False
        try:
            value = json.loads(message)
        except (TypeError, ValueError):
            return False
        if not isinstance(value, dict) or value.get("connection_id") != connection_id:
            return False
        if value.get("kind") == "page_state":
            fields = self.page_state(value)
            row.update({k: fields[k] for k in ("visibility_state", "has_focus")})
            event = value.get("event")
            self.event(
                row,
                "page_state",
                trigger=event
                if event in ("visibilitychange", "focus", "blur", "pagehide", "pageshow")
                else "unknown",
                **fields,
            )
            return False  # Page-state messages are not heartbeat replies.
        seq = value.get("sequence")
        if value.get("kind") != "heartbeat_ack" or type(seq) is not int:
            return False
        pending = row["pending"].get(seq)
        if (
            pending is None
            or seq <= row["accepted_sequence"]
            or not isinstance(value.get("nonce"), str)
            or not secrets.compare_digest(value["nonce"], pending["nonce"])
            or not 0 <= self.clock() - pending["sent_at"] <= self.timeout
        ):
            self.event(row, "heartbeat_rejected", sequence=seq)
            return False
        row["accepted_sequence"] = seq
        row["rtt_ms"] = (self.clock() - pending["sent_at"]) * 1000
        row["pending"] = {k: v for k, v in row["pending"].items() if k > seq}
        fields = self.page_state(value)
        row.update({k: fields[k] for k in ("visibility_state", "has_focus")})
        self._accepted(row, "server_challenge_v1", **fields)
        return True

    def _accepted(self, row, protocol, **fields):
        now = self.clock()
        gap = None if row["last_ack"] is None else now - row["last_ack"]
        if gap is not None:
            row["max_gap_s"] = max(row["max_gap_s"], gap)
        row.update(last_ack=now, last_ack_unix=self.wall(), protocol=protocol)
        self.touch(row["owner"])
        self.event(
            row,
            "heartbeat_acknowledged",
            protocol=protocol,
            sequence=row["accepted_sequence"],
            rtt_ms=row["rtt_ms"],
            ack_gap_s=gap,
            **fields,
        )

    def close_connection(self, connection_id, *, reason, code=None):
        row = self.connections[connection_id]
        row.update(
            state="closed", closed_at=self.clock(), close_code=code, close_reason=reason[:160]
        )
        self.event(
            row,
            "disconnected",
            close_code=code,
            close_reason=reason[:160],
            last_ack_age_s=None if row["last_ack"] is None else self.clock() - row["last_ack"],
        )

    def client_closed(self, owner, value):
        row = self.connections.get(value["connection_id"])
        if not row or row["owner"] != owner:
            return False
        details = dict(
            code=value["code"],
            reason=value["reason"][:160],
            was_clean=value["was_clean"],
            **self.page_state(value),
        )
        row["client_close"] = details
        self.event(row, "client_close_report", **details)
        return True

    def summary(self, owner):
        rows = [r for r in self.connections.values() if r["owner"] == owner]
        if not rows:
            return dict(protocol="no_connection", active_connections=0, last_ack_age_s=None)
        row = max(
            rows, key=lambda r: r["last_ack"] if r["last_ack"] is not None else r["opened_at"]
        )
        return dict(
            connection_id=row["id"],
            state=row["state"],
            protocol=row["protocol"],
            active_connections=sum(r["state"] == "open" for r in rows),
            last_ack_unix=row["last_ack_unix"],
            last_ack_age_s=None if row["last_ack"] is None else self.clock() - row["last_ack"],
            rtt_ms=row["rtt_ms"],
            max_ack_gap_s=row["max_gap_s"],
            visibility_state=row["visibility_state"],
            has_focus=row["has_focus"],
            close_code=row["close_code"],
            close_reason=row["close_reason"],
            client_close=row["client_close"],
        )

    def close(self):
        self.handler.close()
        self.logger.removeHandler(self.handler)
