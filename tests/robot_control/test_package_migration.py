"""Exercise the relocated package across the snapshot and native process boundaries."""
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

from robot_control.common import CONTROL_ROOT, ROOT, new_run, source_manifest


class PackageMigrationTests(unittest.TestCase):
    def test_source_snapshot_preserves_package_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            run = new_run(Path(folder) / 'run', 'migration-test')
            source = run / 'source'
            for relative in source_manifest():
                self.assertEqual((source / relative).read_bytes(), (ROOT / relative).read_bytes())
            self.assertTrue((source / 'config/planning.json').is_file())
            self.assertTrue((source / 'robot_control/diagnostics/measured_hold_probe.py').is_file())
            self.assertTrue((source / 'tests/robot_control/test_planner_loop.py').is_file())

    def test_native_qualification_helper_resolves_from_another_working_directory(self):
        # Missing evidence is rejected before native ROS initialization. No control
        # operation is submitted to the socket or to the robot.
        with tempfile.TemporaryDirectory() as folder:
            parent, child = socket.socketpair()
            try:
                process = subprocess.run(
                    [str(CONTROL_ROOT / 'run_native_loop.sh'),
                     '--fd', str(child.fileno()), '--domain', '24', '--namespace', '/tj',
                     '--profile', str(ROOT / 'config/closed_loop_v1.json'),
                     '--qualification', str(Path(folder) / 'missing-evidence.json'),
                     '--output', folder],
                    pass_fds=(child.fileno(),), cwd=folder, capture_output=True,
                    text=True, timeout=20,
                )
                self.assertNotEqual(process.returncode, 0)
                self.assertIn('qualification_missing', process.stderr)
                self.assertIn('production_qualification_rejected', process.stderr)
                self.assertNotIn('ModuleNotFoundError', process.stderr)
            finally:
                parent.close()
                child.close()


if __name__ == '__main__':
    unittest.main()
