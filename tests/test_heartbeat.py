import json

from forcevla_web.heartbeat import Heartbeats


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def setup(tmp_path):
    clock = Clock()
    seen = {}
    tracker = Heartbeats(
        tmp_path,
        lambda owner: seen.update({owner: clock()}),
        clock=clock,
        wall=lambda: 1000 + clock(),
    )
    return tracker, clock, seen


def reply(challenge, **extra):
    return json.dumps(
        dict(
            kind="heartbeat_ack",
            connection_id=challenge["connection_id"],
            sequence=challenge["sequence"],
            nonce=challenge["nonce"],
            visibility_state="visible",
            has_focus=True,
            **extra,
        )
    )


def test_challenge_response_records_server_measured_latency(tmp_path):
    hb, clock, seen = setup(tmp_path)
    try:
        conn = hb.open("owner", peer="test")
        challenge = hb.challenge(conn)
        assert not seen  # Sending status/challenges does not renew liveness.
        clock.now += 0.125
        assert hb.receive(conn, reply(challenge))
        assert seen["owner"] == clock.now
        health = hb.summary("owner")
        assert health["protocol"] == "server_challenge_v1"
        assert health["rtt_ms"] == 125
        assert health["visibility_state"] == "visible"
        events = [
            json.loads(row)
            for row in (tmp_path / "connection-events.jsonl").read_text().splitlines()
        ]
        assert [e["event"] for e in events] == [
            "connected",
            "challenge_sent",
            "heartbeat_acknowledged",
        ]
        assert all("owner" not in e and "nonce" not in e for e in events)
    finally:
        hb.close()


def test_duplicates_wrong_connection_and_stale_replies_cannot_renew(tmp_path):
    hb, clock, seen = setup(tmp_path)
    try:
        a, b = hb.open("a"), hb.open("b")
        challenge = hb.challenge(a)
        assert not hb.receive(b, reply(challenge))
        assert hb.receive(a, reply(challenge))
        ack_at = seen["a"]
        clock.now += 1
        assert not hb.receive(a, reply(challenge))
        assert seen["a"] == ack_at
        next_challenge = hb.challenge(a)
        clock.now += 5.01
        assert not hb.receive(a, reply(next_challenge))
        assert seen["a"] == ack_at
        assert not hb.receive(a, "ping")  # No downgrade after protocol upgrade.
    finally:
        hb.close()


def test_background_visibility_is_diagnostic_not_liveness(tmp_path):
    hb, clock, seen = setup(tmp_path)
    try:
        conn = hb.open("owner")
        challenge = hb.challenge(conn)
        data = json.loads(reply(challenge))
        data.update(visibility_state="hidden", has_focus=False)
        assert hb.receive(conn, json.dumps(data))
        clock.now += 1
        assert not hb.receive(
            conn,
            json.dumps(
                dict(
                    kind="page_state",
                    connection_id=conn,
                    event="visibilitychange",
                    visibility_state="visible",
                    has_focus=True,
                )
            ),
        )
        assert seen["owner"] == 100.0
        assert hb.summary("owner")["visibility_state"] == "visible"
    finally:
        hb.close()


def test_multiple_tabs_and_close_diagnostics_are_owner_scoped(tmp_path):
    hb, clock, seen = setup(tmp_path)
    try:
        a, b = hb.open("owner"), hb.open("owner")
        assert hb.receive(a, reply(hb.challenge(a)))
        clock.now += 1
        assert hb.receive(b, reply(hb.challenge(b)))
        hb.close_connection(a, reason="peer_disconnected", code=1001)
        assert hb.summary("owner")["active_connections"] == 1
        assert hb.summary("owner")["connection_id"] == b
        report = dict(
            connection_id=a,
            code=1001,
            reason="page navigation",
            was_clean=True,
            visibility_state="hidden",
            has_focus=False,
        )
        assert not hb.client_closed("another-owner", report)
        assert hb.client_closed("owner", report)
        hb.close_connection(b, reason="status_send_timeout", code=1011)
        assert hb.summary("owner")["active_connections"] == 0
        assert hb.summary("owner")["close_code"] == 1011
    finally:
        hb.close()


def test_old_page_compatibility_is_labeled_and_pending_storage_is_bounded(tmp_path):
    hb, clock, seen = setup(tmp_path)
    try:
        conn = hb.open("owner")
        assert hb.receive(conn, "ping")
        assert hb.summary("owner")["protocol"] == "legacy_client_ping"
        for _ in range(30):
            clock.now += 1
            hb.challenge(conn)
        assert len(hb.connections[conn]["pending"]) <= 6
        assert seen["owner"] == 100.0
        challenge = hb.challenge(conn)
        assert hb.receive(conn, reply(challenge))
        assert hb.summary("owner")["protocol"] == "server_challenge_v1"
    finally:
        hb.close()
