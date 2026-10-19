"""Run with the ROS venv; these tests never initialize ROS or touch hardware."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

av = pytest.importorskip("av")
pytest.importorskip("cv2")
pytest.importorskip("rclpy")

from forcevla_web.collector import TileDecoder, split_tile  # noqa: E402
from forcevla_web.observation import CAMERAS  # noqa: E402


def test_quadrants_rgb_padding_and_shape_guard():
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    colors = ((240, 10, 20), (20, 235, 10), (10, 20, 240), (230, 230, 20))
    for color, (y, x) in zip(colors, ((0, 0), (0, 960), (540, 0), (540, 960))):
        rgb[y : y + 540, x : x + 960] = color
    result = split_tile(rgb)
    for key, color in zip(CAMERAS, colors):
        assert result[key].shape == (480, 640, 3)
        np.testing.assert_allclose(result[key][240, 320], color, atol=2)
        assert result[key][:55].max() <= 2
        assert result[key][425:].max() <= 2
    with pytest.raises(ValueError, match="Unexpected tile shape"):
        split_tile(np.zeros((480, 640, 3), np.uint8))


def test_jpeg_decoder_preserves_source_time_and_throttles():
    rgb = np.full((1080, 1920, 3), 90, np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG")
    msg = SimpleNamespace(
        format="jpeg",
        data=buffer.getvalue(),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=200000000)),
    )
    decoder = TileDecoder((1080, 1920, 3))
    frames, receipt, source = decoder.decode(msg, 10.0)
    assert receipt == 10 and source == 123.2
    assert set(frames) == set(CAMERAS)
    assert decoder.decode(msg, 10.1) is None
    assert decoder.decode(msg, 10.3) is not None


def test_h264_keeps_intervening_predictive_frames():
    from fractions import Fraction

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = 128
    encoder.height = 72
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "g": "60"}
    decoder = TileDecoder((72, 128, 3))
    outputs = []
    for i in range(20):
        pixels = np.full((72, 128, 3), i * 10, np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = i
        for packet in encoder.encode(frame):
            ns = 10**9 + i * 33333333
            msg = SimpleNamespace(
                format="h264",
                data=bytes(packet),
                header=SimpleNamespace(stamp=SimpleNamespace(sec=ns // 10**9, nanosec=ns % 10**9)),
            )
            result = decoder.decode(msg, 10 + i / 30)
            if result:
                outputs.append((i, result))
    assert len(outputs) >= 3
    assert outputs[-1][0] >= 12
    assert outputs[-1][1][0]["image"][240, 320].mean() > 100


def test_configurable_output_cadence_preserves_source_age():
    rgb = np.full((1080, 1920, 3), 90, np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG")
    msg = SimpleNamespace(
        format="jpeg",
        data=buffer.getvalue(),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=0)),
    )
    decoder = TileDecoder((1080, 1920, 3), output_period_s=0.09)
    assert decoder.decode(msg, 10.0)[2] == 123.0
    assert decoder.decode(msg, 10.05) is None
    msg.header.stamp.nanosec = 100000000
    result = decoder.decode(msg, 10.1)
    assert result[1:] == (10.1, 123.1)
    for value in (0, -1, float("nan"), True):
        with pytest.raises(ValueError):
            TileDecoder((1080, 1920, 3), value)


def test_latest_decoded_frame_replaces_work_without_refreshing_timestamps():
    from forcevla_web.collector import LatestDecodedFrame

    mailbox = LatestDecodedFrame()
    old_frame = object()
    latest_frame = object()
    mailbox.put((old_frame, 10.0, 100.0))
    mailbox.put((latest_frame, 10.1, 100.1))
    assert mailbox.replaced == 1
    assert mailbox.take() == (latest_frame, 10.1, 100.1)
    assert mailbox.take(timeout=0) is None


def test_h264_receive_lane_does_not_render_or_drop_predictive_packets():
    from fractions import Fraction

    from forcevla_web.collector import LatestDecodedFrame

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = 128
    encoder.height = 72
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "g": "60"}
    decoder = TileDecoder((72, 128, 3), output_period_s=.09)
    mailbox = LatestDecodedFrame()
    # Rendering is deliberately deferred until all compressed frames have arrived.
    for i in range(30):
        frame = av.VideoFrame.from_ndarray(np.full((72, 128, 3), i*5, np.uint8), format="rgb24")
        frame.pts = i
        for packet in encoder.encode(frame):
            ns = 10**9 + i*33333333
            msg = SimpleNamespace(format="h264", data=bytes(packet), header=SimpleNamespace(
                stamp=SimpleNamespace(sec=ns//10**9, nanosec=ns%10**9)))
            decoded = decoder.decode_frame(msg, 10+i/30)
            if decoded:
                mailbox.put(decoded)
    assert decoder.stats["outputs"] == 0
    assert decoder.stats["packets"] == 30
    assert decoder.stats["frames"] >= 29
    result = decoder.render_frame(mailbox.take())
    assert result[1] > 10.8
    assert result[2] > 1.8
    assert result[0]["image"][240, 320].mean() > 125
    assert mailbox.replaced >= 27
