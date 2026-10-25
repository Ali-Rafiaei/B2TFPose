#!/usr/bin/env python3
"""
live_demo_client.py
Laptop-side client for the B2TF RGBD live pose estimation demo.

Connects to the Zivid camera, waits for the presenter to press ENTER, captures
an RGB-D frame, sends it to the server over Tailscale, and displays the
returned pose-overlay image.

Usage:
    python live_demo_client.py --server http://<server-tailscale-ip>:5000
"""

import argparse
import io

import numpy as np
import requests
from PIL import Image

from zivid import Application
from zivid.presets import categories


def convert_zivid_frame_to_rgb_depth(frame):
    """Extract RGB (uint8 H×W×3) and depth (float32 H×W, mm) from a Zivid frame."""
    point_cloud = frame.point_cloud()
    depth = point_cloud.copy_data("z")
    depth[np.isnan(depth)] = 0
    rgba = point_cloud.copy_data("rgba")
    rgb = rgba[:, :, :3]
    return rgb, depth


def send_frame(server_url, rgb, depth, timeout):
    """POST an RGB-D frame to the server. Returns the overlay as a PIL.Image, or raises."""
    rgb_buf = io.BytesIO()
    Image.fromarray(rgb).save(rgb_buf, format="JPEG", quality=95)
    rgb_buf.seek(0)

    depth_buf = io.BytesIO()
    np.save(depth_buf, depth)
    depth_buf.seek(0)

    resp = requests.post(
        f"{server_url}/estimate",
        files={
            "rgb":   ("frame.jpg", rgb_buf, "image/jpeg"),
            "depth": ("frame.npy", depth_buf, "application/octet-stream"),
        },
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Server returned {resp.status_code}: {resp.text}")

    return Image.open(io.BytesIO(resp.content))


def main():
    parser = argparse.ArgumentParser(description="B2TF RGBD live demo client")
    parser.add_argument("--server", required=True,
                        help="Server URL over Tailscale, e.g. http://100.x.x.x:5000")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="HTTP request timeout in seconds (default: 120)")
    args = parser.parse_args()
    server_url = args.server.rstrip("/")
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    print("Connecting to Zivid camera...")
    app = Application()
    with app.connect_camera() as camera:
        print(f"Here are the camera presets for your model ({camera.info.model}):")
        for category in categories(camera.info.model):
            print(f"  {category.name}:")
            for preset in category.presets:
                print(f"    {preset.name}")
        print()
        chosen_preset = categories(camera.info.model)[1].presets[0]
        print("Camera connected and ready.")
        print()
        print("#" * 60)
        print("Press ENTER to capture a frame and estimate poses.")
        print("Press 'q' then ENTER to quit.")
        print("#" * 60)

        while True:
            key = input().strip().lower()
            if key == "q":
                print("Exiting.")
                break

            try:
                print("Capturing frame...")
                with camera.capture(chosen_preset.settings) as frame:
                    rgb, depth = convert_zivid_frame_to_rgb_depth(frame)

                print("Displaying captured image.")
                Image.fromarray(rgb).show()

                print(f"Sending to server ({server_url}) — this can take several seconds...")
                overlay = send_frame(server_url, rgb, depth, timeout=args.timeout)

                print("Displaying result.")
                overlay.show()

            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network error: {e}")
            except RuntimeError as e:
                print(f"[ERROR] {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")

            print()
            print("Press ENTER to capture another frame, or 'q' then ENTER to quit.")


if __name__ == "__main__":
    main()
