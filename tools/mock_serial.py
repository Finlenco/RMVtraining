#!/usr/bin/env python3
"""Run RMVtraining against a pseudo-terminal that simulates the controller."""

from __future__ import annotations

import argparse
import errno
import math
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import time
import tty


FRAME_HEAD = 0x71
FRAME_TAIL = 0x4C
FRAME_SIZE = 64
FRAME_HEAD_BYTES = bytes((FRAME_HEAD,))
FRAME_STRUCT = struct.Struct("<BfffBBBfffffffBffffHB")
FIELD_NAMES = (
    "head",
    "yaw",
    "pitch",
    "roll",
    "status",
    "is_far",
    "armor_flag",
    "latency",
    "bias",
    "distance",
    "pitch_offset",
    "coo_x",
    "coo_y",
    "wheel_w",
    "fire_allowance",
    "target_yaw",
    "target_pitch",
    "yaw_vel",
    "pitch_vel",
    "crc",
    "tail",
)

assert FRAME_STRUCT.size == FRAME_SIZE


def encode_controller_frame(
    status: int = 0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    bias: float = 0.0,
    is_far: int = 0,
) -> bytes:
    """Build one controller-to-vision frame using the project's 64-byte layout."""
    return FRAME_STRUCT.pack(
        FRAME_HEAD,
        yaw,
        pitch,
        roll,
        status,
        is_far,
        0,  # armor_flag
        0.0,  # latency
        bias,
        0.0,  # distance
        0.0,  # pitch_offset
        0.0,  # coo_x
        0.0,  # coo_y
        0.0,  # wheel_w
        0,  # fire_allowance
        yaw,
        pitch,
        0.0,  # yaw_vel
        0.0,  # pitch_vel
        0,  # crc / EKF state
        FRAME_TAIL,
    )


def decode_frame(frame: bytes) -> dict[str, int | float]:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"frame must be {FRAME_SIZE} bytes, got {len(frame)}")
    values = FRAME_STRUCT.unpack(frame)
    if values[0] != FRAME_HEAD or values[-1] != FRAME_TAIL:
        raise ValueError("invalid frame markers")
    return dict(zip(FIELD_NAMES, values))


class FrameParser:
    """Collect fixed-size frames from arbitrarily chunked PTY reads."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer.extend(data)
        frames: list[bytes] = []
        while True:
            head = self.buffer.find(FRAME_HEAD_BYTES)
            if head < 0:
                self.buffer.clear()
                break
            if head:
                del self.buffer[:head]
            if len(self.buffer) < FRAME_SIZE:
                break
            if self.buffer[FRAME_SIZE - 1] != FRAME_TAIL:
                del self.buffer[0]
                continue
            frames.append(bytes(self.buffer[:FRAME_SIZE]))
            del self.buffer[:FRAME_SIZE]
        return frames


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _web_port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用伪终端模拟电控板串口，并显示视觉程序回包。"
    )
    parser.add_argument("--app", default="./build-rmvtraining/rmv_training",
                        help="rmv_training 可执行文件路径")
    parser.add_argument("--camera-config", default="config/camera.yaml",
                        help="相机标定 YAML 路径")
    parser.add_argument("--baud", type=int, default=115200,
                        help="传给程序的波特率参数（PTY 本身不受波特率限制）")
    parser.add_argument("--web-port", type=_web_port,
                        help="可选 Web Debug 端口；不填时使用程序默认值")
    parser.add_argument("--status", type=int, choices=(0, 5), default=0,
                        help="模拟电控板 status：0 红方，5 蓝方")
    parser.add_argument("--rate-hz", type=_positive_float, default=50.0,
                        help="控制器输入帧频率，默认 50 Hz")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行秒数；0 表示按 Ctrl+C 结束")
    parser.add_argument("--yaw", type=_finite_float, default=0.0,
                        help="模拟云台 yaw，单位 rad")
    parser.add_argument("--pitch", type=_finite_float, default=0.0,
                        help="模拟云台 pitch，单位 rad")
    parser.add_argument("--roll", type=_finite_float, default=0.0,
                        help="模拟云台 roll，单位 rad")
    parser.add_argument("--bias", type=_finite_float, default=0.0,
                        help="模拟预测时间偏置，单位 s")
    return parser


def _request_stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)


def run(args: argparse.Namespace) -> int:
    if args.baud <= 0:
        print("波特率必须大于零", file=sys.stderr)
        return 2
    if not math.isfinite(args.duration) or args.duration < 0:
        print("运行秒数必须是大于等于零的有限数", file=sys.stderr)
        return 2

    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)
    # 提前关闭 PTY 回显，避免初始化程序前写入的输入帧被当成视觉回包。
    tty.setraw(slave_fd)

    command = [args.app, slave_name, str(args.baud), args.camera_config]
    if args.web_port is not None:
        command.append(str(args.web_port))

    try:
        process = subprocess.Popen(command)
    except OSError as exc:
        os.close(master_fd)
        os.close(slave_fd)
        print(f"无法启动视觉程序 {args.app}: {exc}", file=sys.stderr)
        return 127
    finally:
        # 子进程会自行打开 slave_name；父进程只保留 master 端。
        try:
            os.close(slave_fd)
        except OSError:
            pass

    interval = 1.0 / args.rate_hz
    start = time.monotonic()
    next_send = start
    next_report = start + 1.0
    stop_deadline: float | None = None
    kill_deadline: float | None = None
    stopping = False
    sent_frames = 0
    received_frames = 0
    reported_sent = 0
    reported_received = 0
    latest_output: dict[str, int | float] | None = None
    output_parser = FrameParser()
    controller_frame = encode_controller_frame(
        status=args.status,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        bias=args.bias,
    )

    print(f"[模拟串口] 已启动，PTY={slave_name}，输入频率={args.rate_hz:g} Hz，status={args.status}",
          flush=True)
    print("[模拟串口] 按 Ctrl+C 停止；只使用伪终端，不会打开实体串口。", flush=True)

    try:
        while process.poll() is None:
            now = time.monotonic()
            if args.duration > 0 and not stopping and now - start >= args.duration:
                _request_stop(process)
                stopping = True
                stop_deadline = now + 5.0
            if stop_deadline is not None and now >= stop_deadline and process.poll() is None:
                process.terminate()
                stop_deadline = None
                kill_deadline = now + 2.0
            if kill_deadline is not None and now >= kill_deadline and process.poll() is None:
                process.kill()
                kill_deadline = None

            if not stopping and now >= next_send:
                try:
                    written = os.write(master_fd, controller_frame)
                    if written == FRAME_SIZE:
                        sent_frames += 1
                except OSError as exc:
                    if exc.errno not in (errno.EIO, errno.EAGAIN, errno.EWOULDBLOCK):
                        raise
                next_send = max(next_send + interval, now + interval)

            until_send = next_send - time.monotonic() if not stopping else 0.05
            timeout = max(0.0, min(0.05, until_send))
            readable, _, _ = select.select([master_fd], [], [], timeout)
            if readable:
                try:
                    data = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno in (errno.EIO, errno.EAGAIN, errno.EWOULDBLOCK):
                        data = b""
                    else:
                        raise
                for frame in output_parser.feed(data):
                    try:
                        latest_output = decode_frame(frame)
                    except ValueError:
                        continue
                    received_frames += 1

            now = time.monotonic()
            if now >= next_report:
                rx_delta = sent_frames - reported_sent
                tx_delta = received_frames - reported_received
                print(
                    f"[RX 电控→视觉] +{rx_delta} 帧，status={args.status}，"
                    f"yaw={args.yaw:.3f} rad，pitch={args.pitch:.3f} rad，bias={args.bias:.3f} s",
                    flush=True,
                )
                if latest_output is None:
                    tx_summary = "尚未收到视觉回包"
                else:
                    tx_summary = (
                        f"目标={int(latest_output['armor_flag'])}，"
                        f"距离={float(latest_output['distance']):.2f} m，"
                        f"指令角=({float(latest_output['target_yaw']):.3f}, "
                        f"{float(latest_output['target_pitch']):.3f}) rad，"
                        f"开火={int(latest_output['fire_allowance'])}，"
                        f"状态={int(latest_output['crc'])}，"
                        f"延迟={float(latest_output['latency']):.1f} ms"
                    )
                print(f"[TX 视觉→电控] +{tx_delta} 帧，{tx_summary}", flush=True)
                reported_sent = sent_frames
                reported_received = received_frames
                next_report = now + 1.0
    except KeyboardInterrupt:
        print("\n[模拟串口] 正在请求视觉程序退出……", flush=True)
        _request_stop(process)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
    finally:
        if process.poll() is None:
            _request_stop(process)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        os.close(master_fd)

    print(f"[模拟串口] 结束：发送控制帧 {sent_frames} 帧，收到视觉回包 {received_frames} 帧。",
          flush=True)
    return process.returncode if process.returncode is not None else 0


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
