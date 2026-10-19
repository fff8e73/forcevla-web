"""Compare same-host DDS transport without changing the camera publisher."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import av
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=20)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rclpy.init()
node = rclpy.create_node("forcevla_transport_probe")
codec = av.CodecContext.create("h264", "r")
codec.thread_count = 2
stats = {"packets": 0, "frames": 0, "decode_errors": 0, "stamp_gaps": 0, "no_frame": 0}
last = None
first_decoded = None
after_frames = 0
after_packets = 0


def receive(msg):
    global last, first_decoded, after_packets, after_frames
    stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
    if last and stamp - last > 60000000:
        stats["stamp_gaps"] += 1
    last = stamp
    stats["packets"] += 1
    if first_decoded is not None:
        after_packets += 1
    try:
        frames = codec.decode(av.Packet(bytes(msg.data)))
    except av.FFmpegError:
        stats["decode_errors"] += 1
        return
    stats["frames"] += len(frames)
    if not frames:
        stats["no_frame"] += 1
    if frames and first_decoded is None:
        first_decoded = time.monotonic()
    elif first_decoded is not None:
        after_frames += len(frames)


sub = node.create_subscription(
    CompressedImage,
    "/quad_tile/compressed_undistorted",
    receive,
    QoSProfile(depth=32, reliability=ReliabilityPolicy.BEST_EFFORT),
)
start = time.monotonic()
try:
    while time.monotonic() - start < args.duration:
        rclpy.spin_once(node, timeout_sec=0.02)
    stats["subscriber_gids"] = [
        bytes(x.endpoint_gid).hex()
        for x in node.get_subscriptions_info_by_topic("/quad_tile/compressed_undistorted")
        if x.node_name == "forcevla_transport_probe"
    ]
finally:
    inodes = set()
    for f in Path("/proc/self/fd").iterdir():
        try:
            link = str(f.readlink())
        except OSError:
            continue
        m = re.fullmatch(r"socket:\[(\d+)\]", link)
        if m:
            inodes.add(m[1])
    stats["udp_sockets"] = []
    for row in Path("/proc/net/udp").read_text().splitlines()[1:]:
        fields = row.split()
        if fields[9] in inodes:
            stats["udp_sockets"].append({"local": fields[1], "drops": int(fields[-1])})
    stats["shm_maps"] = [
        row.split()[-1]
        for row in Path("/proc/self/maps").read_text().splitlines()
        if "/dev/shm/fastrtps" in row
    ]
    node.destroy_node()
    rclpy.shutdown()
stats.update(
    duration_s=time.monotonic() - start,
    packets_after_first_decoded=after_packets,
    frames_after_first_decoded=after_frames,
    profile=os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"),
)
Path(args.output).write_text(json.dumps(stats, indent=2) + "\n")
print(json.dumps(stats, indent=2))
