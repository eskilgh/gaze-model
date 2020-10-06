import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from pprint import pprint
import pandas as pd


class BrownianModel:
    # Delta value for time in ms
    T_INC = 8
    D = 0.00005
    # Saccadic duration in ms
    SACC_DUR = 8 * 5
    # Saccade velocity
    SACC_VEL = 0.0150
    rng = default_rng()

    def __init__(
        self,
        width: int = 1440,
        height: int = 1600,
        state_ratio: int = 40,
        x_start=None,
        y_start=None,
    ):
        self.animator = self.Animator(self)
        self.state_ratio = state_ratio
        self.crt_time = 0

        x_start = x_start if x_start else 0.5
        y_start = y_start if y_start else 0.5

        self.entries = {
            "time": [self.crt_time],
            "x": [x_start],
            "y": [y_start],
            "state": ["FIXATION"],
        }

    def generate(self, duration: int = 2000):
        while self.crt_time < duration:
            self.add_fixation(200)
            self.add_saccade(40)

    def add_fixation(self, duration):
        start_time = self.crt_time
        while self.crt_time < (duration + start_time) and self.crt_time < 2000:
            x, y = self.generate_brownian_step()
            self.entries["x"].append(x)
            self.entries["y"].append(y)
            self.crt_time = self.entries["time"][-1] + self.T_INC
            self.entries["time"].append(self.crt_time)
            self.entries["state"].append("FIXATION")

    def generate_brownian_step(self):
        x_prev = self.entries["x"][-1]
        y_prev = self.entries["y"][-1]

        x = -1
        y = -1
        while not self.is_within_bounds(x):
            x = x_prev + self.rng.normal() * np.sqrt(self.D * self.T_INC)
        while not self.is_within_bounds(y):
            y = y_prev + self.rng.normal() * np.sqrt(self.D * self.T_INC)
        return x, y

    def add_saccade(self, duration):
        x_prev = self.entries["x"][-1]
        y_prev = self.entries["y"][-1]

        x_dest = -1
        y_dest = -1
        while x_dest < 0.05 or x_dest > 0.95:
            x_dest = x_prev + self.SACC_VEL * duration * np.cos(
                self.rng.random() * 2 * np.pi
            )
        while y_dest < 0.05 or y_dest > 0.95:
            y_dest = y_prev + self.SACC_VEL * duration * np.cos(
                self.rng.random() * 2 * np.pi
            )

        nr_of_steps = duration // self.T_INC
        x_steps = self.split_distance(x_prev, x_dest, nr_of_steps)
        y_steps = self.split_distance(y_prev, y_dest, nr_of_steps)
        timestamps = [self.crt_time + i * self.T_INC for i in range(nr_of_steps)]

        self.entries["x"].extend(x_steps)
        self.entries["y"].extend(y_steps)
        self.entries["time"].extend(timestamps)
        self.entries["state"].extend(["SACCADE"] * nr_of_steps)

    def split_distance(self, start, end, nr_of_steps):
        step_length = (end - start) / nr_of_steps
        steps = [start + step_length * i for i in range(nr_of_steps)]
        # Hard code last step
        steps[-1] = end
        return steps

    @staticmethod
    def is_within_bounds(val, lower_bound=0.0, upper_bound=1.0):
        return lower_bound <= val <= upper_bound

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

    class Animator:
        def __init__(self, parent):
            self.parent = parent

        def animated_plot(self):
            assert len(self.parent.entries["x"]) > 2

            fig, ax = plt.subplots()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            (self.plot,) = ax.plot(0, 0)
            self.xdata = self.parent.entries["x"][0:2]
            self.ydata = self.parent.entries["y"][0:2]
            self.plot.set_xdata(self.xdata)
            self.plot.set_ydata(self.ydata)

            animation = FuncAnimation(
                fig,
                func=self.animation_frame,
                frames=zip(self.parent.entries["x"], self.parent.entries["y"]),
                interval=self.parent.T_INC,
                repeat=False,
            )
            plt.show()

        def animation_frame(self, coordinate):
            x, y = coordinate
            self.xdata.pop(0)
            self.ydata.pop(0)

            self.xdata.append(x)
            self.ydata.append(y)
            self.plot.set_xdata(self.xdata)
            self.plot.set_ydata(self.ydata)

            return self.plot


model = BrownianModel()
model.generate()
model.plot()
model.animator.animated_plot()
