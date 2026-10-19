"""Inbound heartbeat processing must continue while status sending blocks."""
import asyncio
import threading

from starlette.websockets import WebSocket

from test_prediction import system, wait_for


def test_heartbeats_are_received_during_slow_status_send(system, monkeypatch):
    client, manager, _ = system
    original = WebSocket.send_json
    blocked = threading.Event()
    calls = 0

    async def delayed(self, data, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            blocked.set()
            await asyncio.sleep(1.5)
        return await original(self, data, *args, **kwargs)

    monkeypatch.setattr(WebSocket, 'send_json', delayed)
    with client.websocket_connect('/api/v1/events') as ws:
        data = ws.receive_json()
        challenge = data['heartbeat']
        assert blocked.wait(2)
        ws.send_json(dict(kind='heartbeat_ack', **challenge,
                          visibility_state='visible', has_focus=True))
        # Receive without waiting for the blocked 1.5 second status send.
        wait_for(lambda: manager.heartbeats.connections[challenge['connection_id']]['last_ack'] is not None,
                 timeout=.7)
