"""Subscriptions and bounded H.264 decoding only; no application publishers."""

from fractions import Fraction
import json
import queue
import threading
import time

import av
import cv2
import numpy as np

from robot_control.observation import Rejected, gripper_values, pose6, live_training_tiles, wrench6


def stamp_ns(stamp):
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
        raise Rejected("invalid_header_stamp")
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class ROSObserver:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.stop_event = threading.Event()
        self.packets = queue.Queue(maxsize=8)
        self.frames = queue.Queue(maxsize=1)
        self.reset_decoder = threading.Event()
        self.threads = []
        self.failure = None
        self.context = self.node = self.executor = None
        self.last_packet_ns = 0

    def start(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import String

        self.context = Context()
        rclpy.init(
            args=[],
            context=self.context,
            domain_id=self.config["ros_domain_id"],
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self.node = Node(
            "forcevla_observe_readonly",
            context=self.context,
            use_global_arguments=False,
            enable_rosout=False,
            start_parameter_services=False,
            enable_logger_service=False,
        )
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for key, topic in self.config["topics"].items():
            cls = (
                String
                if key == "gripper"
                else PoseStamped
                if key.endswith("pose")
                else WrenchStamped
            )
            self.node.create_subscription(cls, topic, self.numeric_callback(key), qos)
        if self.config.get("require_joints"):
            self.node.create_subscription(
                JointState, "/tj/joint_states", self.numeric_callback("joints"), qos
            )
        self.node.create_subscription(
            CompressedImage, self.config["camera_topic"], self.on_camera, qos
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        for name, target in [
            ("forcevla-ros-receive", self.spin),
            ("forcevla-camera-decode", self.decode),
            ("forcevla-image-transform", self.transform),
        ]:
            thread = threading.Thread(name=name, target=self.guard(target), daemon=True)
            self.threads.append(thread)
            thread.start()
        return self

    def guard(self, target):
        def run():
            try:
                target()
            except Exception as exc:
                self.failure = repr(exc)
                self.stop_event.set()

        return run

    def spin(self):
        while not self.stop_event.is_set() and self.context.ok():
            self.executor.spin_once(timeout_sec=0.1)

    def numeric_callback(self, key):
        def receive(message):
            wall, mono = time.time_ns(), time.monotonic_ns()
            try:
                metadata = {}
                if key == "joints":
                    names = [
                        f"Joint{i}_{side}" for side in ["L", "R"] for i in range(1, 8)
                    ]
                    if len(message.name) != len(message.position) or len(
                        set(message.name)
                    ) != len(message.name):
                        raise Rejected("invalid_joint_names")
                    mapping = dict(zip(message.name, message.position))
                    value = np.asarray(
                        [mapping[name] for name in names], dtype=np.float64
                    )
                    if not np.isfinite(value).all():
                        raise Rejected("nonfinite_joints")
                    source = stamp_ns(message.header.stamp)
                    metadata["joint_names"] = names
                    if self.config.get("require_joint_velocity"):
                        velocities = list(message.velocity)
                        if len(velocities) != len(message.name):
                            raise Rejected("missing_joint_velocity")
                        mapped = dict(zip(message.name, velocities))
                        ordered = np.asarray(
                            [mapped[name] for name in names], dtype=float
                        )
                        if not np.isfinite(ordered).all():
                            raise Rejected("nonfinite_joint_velocity")
                        metadata["velocities_rad_s"] = ordered.tolist()
                elif key == "gripper":
                    obj = json.loads(message.data)
                    source = obj["sample_time_ns"]
                    value = gripper_values(obj, self.config["expected_calibration"])
                    metadata = {
                        k: obj.get(k)
                        for k in [
                            "calibration",
                            "reserved",
                            "control_source",
                            "applied_target",
                            "healthy",
                            "errors",
                        ]
                    }
                else:
                    source = stamp_ns(message.header.stamp)
                    metadata["frame_id"] = message.header.frame_id
                    if key.endswith("pose"):
                        p, q = message.pose.position, message.pose.orientation
                        value, singular = pose6(
                            [p.x, p.y, p.z],
                            [q.x, q.y, q.z, q.w],
                            message.header.frame_id,
                        )
                        metadata["euler_singularity_warning"] = singular
                    else:
                        f, t = message.wrench.force, message.wrench.torque
                        value = wrench6(
                            [f.x, f.y, f.z],
                            [t.x, t.y, t.z],
                            message.header.frame_id,
                            "flange_L" if key.startswith("left") else "flange_R",
                        )
                self.store.update(
                    key, value, source, metadata, receipt_ns=wall, receipt_mono_ns=mono
                )
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                self.store.reject(key, str(exc))

        return receive

    def on_camera(self, message):
        wall, mono = time.time_ns(), time.monotonic_ns()
        try:
            source = stamp_ns(message.header.stamp)
            if message.header.frame_id != self.config["camera_frame"]:
                raise Rejected("unexpected_camera_frame:" + message.header.frame_id)
            if source <= self.last_packet_ns:
                raise Rejected("duplicate_or_out_of_order_packet")
            if len(message.data) > 16 * 1024 * 1024:
                raise Rejected("oversize_packet")
            self.last_packet_ns = source
            item = (bytes(message.data), message.format.lower(), source, wall, mono)
            try:
                self.packets.put_nowait(item)
            except queue.Full:
                self.reset_decoder.set()
                while True:
                    try:
                        self.packets.get_nowait()
                    except queue.Empty:
                        break
                self.packets.put_nowait(item)
                raise Rejected("decoder_queue_overflow")
        except (ValueError, TypeError) as exc:
            self.store.reject("camera", str(exc))

    def decode(self):
        decoder = av.CodecContext.create("h264", "r")
        decoder.thread_count = 1
        receipts = {}
        while not self.stop_event.is_set():
            try:
                data, kind, source, wall, mono = self.packets.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.reset_decoder.is_set():
                decoder = av.CodecContext.create("h264", "r")
                decoder.thread_count = 1
                receipts.clear()
                self.reset_decoder.clear()
            try:
                if "h264" in kind:
                    packet = av.Packet(data)
                    packet.pts = packet.dts = source
                    packet.time_base = Fraction(1, 1_000_000_000)
                    receipts[source] = (wall, mono)
                    # Bounded bookkeeping even when waiting for the first IDR frame.
                    if len(receipts) > 120:
                        receipts.pop(next(iter(receipts)))
                    frames = decoder.decode(packet)
                    for frame in frames:
                        if (
                            frame.pts is None
                            or frame.pts not in receipts
                            or frame.pict_type == 3
                        ):
                            raise Rejected("unmatched_or_reordered_h264_frame")
                        received, received_mono = receipts.pop(frame.pts)
                        self.queue_frame(
                            frame, frame.pts, received, received_mono, kind
                        )
                elif "jpeg" in kind or "jpg" in kind:
                    bgr = cv2.imdecode(
                        np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if bgr is None:
                        raise Rejected("invalid_jpeg")
                    self.queue_frame(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), source, wall, mono, kind
                    )
                else:
                    raise Rejected("unsupported_camera_format:" + kind)
            except (av.FFmpegError, ValueError, TypeError) as exc:
                self.store.reject("camera", type(exc).__name__ + ":" + str(exc)[:180])

    def queue_frame(self, frame, source, wall, mono, kind):
        # Decode every compressed packet, but only transform the newest decoded
        # frame. Dropping decoded frames does not break H.264 reference chains.
        item = (frame, source, wall, mono, kind)
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(item)

    def transform(self):
        cv2.setNumThreads(2)
        while not self.stop_event.is_set():
            try:
                frame, source, wall, mono, kind = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                rgb = (
                    frame
                    if isinstance(frame, np.ndarray)
                    else frame.to_ndarray(format="rgb24")
                )
                self.accept_frame(rgb, source, wall, mono, kind)
            except (ValueError, TypeError, av.FFmpegError) as exc:
                self.store.reject("camera", type(exc).__name__ + ":" + str(exc)[:180])

    def accept_frame(self, rgb, source, wall, mono, kind):
        tiles = live_training_tiles(rgb, self.config["camera_decoded_shape"])
        metadata = {
            "format": kind,
            "decoded_shape": list(rgb.shape),
            "decode_and_transform_s": (time.monotonic_ns() - mono) / 1e9,
            "preprocessing": self.config["camera_geometry_policy"] + "-jpeg95-444",
            "camera_transport_equivalence_verified": self.config[
                "camera_transport_equivalence_verified"
            ],
        }
        self.store.update(
            "camera",
            tiles,
            int(source),
            metadata,
            receipt_ns=wall,
            receipt_mono_ns=mono,
        )

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=3)
        if any(t.is_alive() for t in self.threads):
            raise RuntimeError("Observer worker did not stop")
        if self.executor:
            self.executor.shutdown(timeout_sec=1)
        if self.node:
            self.node.destroy_node()
        if self.context and self.context.ok():
            self.context.shutdown()
