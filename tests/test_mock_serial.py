import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "mock_serial.py"
SPEC = importlib.util.spec_from_file_location("mock_serial", MODULE_PATH)
mock_serial = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mock_serial)


class MockSerialProtocolTests(unittest.TestCase):
    def test_controller_frame_uses_the_project_layout(self):
        frame = mock_serial.encode_controller_frame(
            status=5, yaw=0.25, pitch=-0.1, roll=0.02, bias=0.04
        )

        self.assertEqual(len(frame), 64)
        decoded = mock_serial.decode_frame(frame)
        self.assertEqual(decoded["head"], 0x71)
        self.assertEqual(decoded["tail"], 0x4C)
        self.assertEqual(decoded["status"], 5)
        self.assertAlmostEqual(decoded["yaw"], 0.25)
        self.assertAlmostEqual(decoded["pitch"], -0.1)
        self.assertAlmostEqual(decoded["bias"], 0.04)

    def test_parser_handles_partial_frames_and_noise(self):
        first = mock_serial.encode_controller_frame(status=0, yaw=0.1)
        second = mock_serial.encode_controller_frame(status=5, yaw=-0.2)
        parser = mock_serial.FrameParser()

        self.assertEqual(parser.feed(b"noise" + first[:17]), [])
        self.assertEqual(parser.feed(first[17:] + second), [first, second])
        self.assertEqual(mock_serial.decode_frame(second)["status"], 5)


if __name__ == "__main__":
    unittest.main()
