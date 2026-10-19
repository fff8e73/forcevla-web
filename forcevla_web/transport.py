"""Passive DDS diagnostics. Never changes transports or restarts camera services."""

import os
import re
import time
from pathlib import Path


def udp_drops(pid=None):
    pid = os.getpid() if pid is None else pid
    inodes = set()
    for descriptor in Path(f"/proc/{pid}/fd").iterdir():
        try:
            match = re.fullmatch(r"socket:\[(\d+)\]", str(descriptor.readlink()))
        except OSError:
            continue
        if match:
            inodes.add(match[1])
    drops = 0
    for name in ("udp", "udp6"):
        for line in Path(f"/proc/{pid}/net/{name}").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[9] in inodes:
                drops += int(fields[-1])
    return drops


def host_prefix(gid):
    value = bytes(gid)
    return value[:4].hex() if len(value) >= 12 else None


def camera_transport(node, topic):
    publishers = node.get_publishers_info_by_topic(topic)
    readers = [
        x for x in node.get_subscriptions_info_by_topic(topic) if x.node_name == node.get_name()
    ]
    publisher_hosts = sorted({host_prefix(x.endpoint_gid) for x in publishers})
    reader_hosts = sorted({host_prefix(x.endpoint_gid) for x in readers})
    known = bool(publisher_hosts and reader_hosts and None not in publisher_hosts + reader_hosts)
    matched = publisher_hosts == reader_hosts if known else None
    return {
        "checked_mono": time.monotonic(),
        "camera_host_prefixes": publisher_hosts,
        "reader_host_prefixes": reader_hosts,
        "same_host_prefix": matched,
        "udp_drops_total": udp_drops(),
        "warning": (
            "相机与采集服务的网络上下文不一致，可能退回易丢包的UDP；请检查相机连接。"
            if matched is False
            else None
        ),
    }
