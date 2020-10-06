import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from pprint import pprint
import pandas as pd
import plotly.express as px


class GazeModel:
    def __init__(
        self,
        width: int = 1440,
        height: int = 1600,
        velocity: float = 1,
        state_ratio: int = 40,
        x_start=None,
        y_start=None,
    ):
        self.velocity = velocity
        self.state_ratio = state_ratio
        self.state = "FIXATION"

        x_start = x_start if x_start else 0.5
        y_start = y_start if y_start else 0.5

        self.entries = {
            "time": [0],
            "x": [x_start],
            "y": [y_start],
            "state": [self.state],
        }

    def generate(self, steps: int = 300):
        for i in range(1, steps):
            self.state = "SACCADE" if i % 10 == 0 else "FIXATION"

            prev_time = self.entries["time"][-1]
            time = self.generate_step_length()
            x, y = self.next_position(time)
            self.entries["time"].append(prev_time + time)
            self.entries["x"].append(x)
            self.entries["y"].append(y)
            self.entries["state"].append(self.state)

        return self.entries

    def next_position(self, time):
        x_prev = self.entries["x"][-1]
        y_prev = self.entries["y"][-1]

        x = self.velocity * time * np.cos(random.rand() * 2 * np.pi) + x_prev
        while not self.is_within_bounds(x):
            x = self.velocity * time * np.cos(random.rand() * 2 * np.pi) + x_prev

        y = self.velocity * time * np.sin(random.rand() * 2 * np.pi) + y_prev
        while not self.is_within_bounds(y):
            y = self.velocity * time * np.sin(random.rand() * 2 * np.pi) + y_prev

        return x, y

    @staticmethod
    def is_within_bounds(val):
        return 0.0 <= val <= 1.0

    def generate_step_length(self):
        beta = 0.01 if self.state == "FIXATION" else 0.40
        return random.exponential(scale=beta)

    def plot(self):
        df = pd.DataFrame(self.entries)

        ax = plt.gca()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        df.plot(kind="line", x="x", y="y", ax=ax, color="r", legend=None)

        # Overlay separate line plot to differentiate saccades from fixations
        saccades = self.split_saccade_from_df(df)
        for sacc_df in saccades:
            sacc_df.plot(kind="line", x="x", y="y", ax=ax, color="b", legend=None)
        plt.show()

    def split_saccade_from_df(self, df):
        """Returns a list of dataframes consisting of saccade rows with leading row"""
        return [df.loc[i - 1 : i] for i in df[df["state"] == "SACCADE"].index]

    def overlay_on_image(self, img_path):
        # TODO
        return


model = GazeModel()
model.generate()
model.plot()
