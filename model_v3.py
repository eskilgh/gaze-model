import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pprint import pprint
import pandas as pd
from numpy.random import default_rng
from pprint import pprint
from gaze_animator import GazeAnimator
from scipy.stats import expon


plt.rcParams[
    "animation.ffmpeg_path"
] = "/Users/eskilgaarehostad/miniconda3/envs/data/bin/ffmpeg"


class GazeModel:
    def __init__(self, dt=8, width: int = 1440, height: int = 1600):
        self.width = width
        self.height = height

        # dt in ms
        self.dt = dt
        self.D = 3e-6
        # Velocity in normalized coords per ms
        self.v = 4e-3
        self.rng = default_rng()

    def generate(self, total_duration, time=0, x=0.5, y=0.5, state="fixation"):
        data = {"time": [], "x": [], "y": [], "state": []}

        while time < total_duration:
            time_vals, x_vals, y_vals = self.generate_phase(time, x, y, state)
            data["time"].extend(time_vals)
            data["x"].extend(x_vals)
            data["y"].extend(y_vals)
            data["state"].extend([state] * len(time_vals))

            time = data["time"][-1]
            x = data["x"][-1]
            y = data["y"][-1]
            state = "fixation" if state == "saccade" else "saccade"

        return data

    def generate_phase(self, time, x, y, state):
        if state == "fixation":
            return self.generate_fixation(time, x, y)
        elif state == "saccade":
            return self.generate_saccade(time, x, y)
        raise ValueError(
            f'State ""{state}"" not recognized, must be either "fixation" or "saccade".'
        )

    def generate_fixation(self, time, x, y):
        fix_duration = int(self.rng.exponential(80) + 60)
        time_vals = []
        x_vals = []
        y_vals = []

        for i in range(int(fix_duration // self.dt)):
            time += self.dt
            x = self.x_fix(x, self.dt, self.D)
            y = self.y_fix(y, self.dt, self.D)
            time_vals.append(time)
            x_vals.append(x)
            y_vals.append(y)

        assert len(time_vals) == len(x_vals) == len(y_vals)

        return time_vals, x_vals, y_vals

    def x_fix(self, x_prev, dt, D):
        xi_x = self.rng.normal()
        return x_prev + xi_x * np.sqrt(D * dt)

    def y_fix(self, y_prev, dt, D):
        xi_y = self.rng.normal()
        return y_prev + xi_y * np.sqrt(D * dt)

    def generate_saccade(self, time, x, y):
        sacc_duration = np.random.exponential(17) + 8
        time_vals = []
        x_vals = []
        y_vals = []
        phi = None
        while phi is None:
            phi = self.draw_angle(sacc_duration, x, y)

        for i in range(int(sacc_duration // self.dt)):
            time += self.dt
            x = self.x_sacc(x, self.dt, self.v, phi)
            y = self.y_sacc(y, self.dt, self.v, phi)
            time_vals.append(time)
            x_vals.append(x)
            y_vals.append(y)

        return time_vals, x_vals, y_vals

    def x_sacc(self, x_prev, dt, v, phi):
        return x_prev + v * dt * np.cos(phi)

    def y_sacc(self, y_prev, dt, v, phi):
        return y_prev + v * dt * np.sin(phi)

    def draw_angle(self, duration, x, y):
        """
        Draws an angle that does not go out of bounds based on saccade duration and velocity
        :return: Angle in radians (float) if trajectory will not end out of bounds, else None
        """
        angle = self.rng.random() * 2 * np.pi
        x_end = self.x_sacc(x, duration, self.v, angle)
        y_end = self.y_sacc(y, duration, self.v, angle)
        if x_end < 0.1 or x_end > 0.9 or y_end < 0.1 or y_end > 0.9:
            return None
        return angle


if __name__ == "__main__":
    model = GazeModel()
    duration = 10e3
    output = model.generate(duration)
    anim = GazeAnimator(data=output, duration=duration, img="test1.png").animate(
        "Simulated gaze trajectory", show=False
    )
    writer = matplotlib.animation.FFMpegWriter(fps=30)
    anim.save("model_v3.mp4", writer=writer)
