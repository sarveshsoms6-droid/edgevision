"""Generate synthetic traffic video for offline pipeline testing and calibration."""

from pathlib import Path
import cv2
import numpy as np


def create_traffic_video(output_path: str = "sample_data/traffic_sample.mp4", num_frames: int = 120, fps: int = 30):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(p), fourcc, float(fps), (w, h))

    # Road background
    for frame_i in range(num_frames):
        # Asphalt road background
        frame = np.full((h, w, 3), 70, dtype=np.uint8)

        # Sidewalks
        frame[0:60, :] = (150, 150, 150)
        frame[300:360, :] = (150, 150, 150)

        # Lane markings (dashed white lines in middle)
        for x_dash in range(0, w, 40):
            frame[175:180, x_dash:x_dash + 25] = (255, 255, 255)

        # Moving Vehicle 1 (Car: blue rectangle)
        car1_x = int((frame_i * 6) % (w + 100)) - 80
        if 0 <= car1_x < w:
            cv2.rectangle(frame, (car1_x, 100), (min(w, car1_x + 90), 160), (180, 50, 50), -1)
            cv2.rectangle(frame, (car1_x + 15, 110), (min(w, car1_x + 70), 150), (220, 200, 200), -1)
            # Wheels
            cv2.circle(frame, (car1_x + 20, 160), 8, (20, 20, 20), -1)
            cv2.circle(frame, (car1_x + 70, 160), 8, (20, 20, 20), -1)

        # Moving Vehicle 2 (Truck/Bus: yellow rectangle moving right to left)
        bus_x = w - int((frame_i * 4) % (w + 160))
        if 0 <= bus_x < w:
            cv2.rectangle(frame, (max(0, bus_x - 140), 200), (bus_x, 280), (30, 180, 220), -1)
            # Windows
            for win_offset in range(20, 120, 25):
                cv2.rectangle(frame, (max(0, bus_x - win_offset - 20), 210), (bus_x - win_offset, 240), (240, 240, 240), -1)

        # Moving Pedestrian (circle head + body on sidewalk)
        ped_x = int(50 + (frame_i * 2) % 300)
        cv2.circle(frame, (ped_x, 30), 10, (180, 180, 220), -1)
        cv2.rectangle(frame, (ped_x - 8, 40), (ped_x + 8, 58), (50, 120, 50), -1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic traffic video at: {p} ({num_frames} frames, {w}x{h}, {fps} FPS)")


if __name__ == "__main__":
    create_traffic_video()
