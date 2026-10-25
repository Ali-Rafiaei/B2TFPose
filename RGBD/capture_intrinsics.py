#!/usr/bin/env python3
"""
capture_intrinsics.py  —  run this once on the laptop before the demo.

Connects to the Zivid camera, reads the calibrated intrinsics, and prints
the camera_intrinsics line ready to paste into configs/demo_live.yaml on
the server.

Usage (on the laptop):
    python capture_intrinsics.py
"""

from zivid import Application
from zivid.experimental import calibration

app = Application()
with app.connect_camera() as camera:
    m = calibration.intrinsics(camera).camera_matrix
    print(f"fx={m.fx:.6f}  fy={m.fy:.6f}  cx={m.cx:.6f}  cy={m.cy:.6f}")
    print()
    print("Paste this line into configs/demo_live.yaml on the server:")
    print(f'camera_intrinsics: "{m.fx},{m.fy},{m.cx},{m.cy}"')
