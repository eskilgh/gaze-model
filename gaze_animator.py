from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class GazeAnimator:
    def __init__(self, data, duration, fps: int = 30):
        assert len(data["time"]) > 0 and (
            len(data["time"]) == len(data["x"]) == len(data["y"]) > 0
        )
        assert fps <= 30, "Does not support fps above 30."

        self.time = data["time"]
        self.x = data["x"]
        self.y = data["y"]
        self.duration = duration
        self.fps = fps
        self.interval = 1000 // fps

    def split(self):
        nr_of_frames = int(np.rint((self.duration * self.fps) / 10 ** 3))
        return (
            np.array_split(self.time, nr_of_frames),
            np.array_split(self.x, nr_of_frames),
            np.array_split(self.y, nr_of_frames),
        )

    def animate(self):
        time, x, y = self.split()

        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        (plot,) = ax.plot(0, 0)
        plot.set_xdata(x.pop(0))
        plot.set_ydata(y.pop(0))

        animation = FuncAnimation(
            fig,
            func=self.animation_frame,
            frames=zip(x, y),
            fargs=[plot],
            interval=self.interval,
            repeat=False,
        )
        plt.show()
        return animation

    def animation_frame(self, points, *fargs):
        plot = fargs[0]
        x_vals, y_vals = points
        plot.set_xdata(x_vals)
        plot.set_ydata(y_vals)
        pass


if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint

    df = pd.read_csv("einar_siri_fixation.tsv", sep="\t")
    cols_to_keep = [
        "Recording timestamp",
        "Participant name",
        "Recording name",
        "Gaze point X (MCSnorm)",
        "Gaze point Y (MCSnorm)",
        "Presented Media name",
    ]
    df = df[cols_to_keep]
    df = df[
        (df["Presented Media name"] == "test1.png")
        & (df["Participant name"] == "Einar")
    ]
    df_start = df.head(600).dropna()
    timestamps = df_start["Recording timestamp"].values
    duration = (timestamps[-1] - timestamps[0]) // 10 ** 3
    data = {
        "time": timestamps,
        "x": [
            float(x.replace(",", "."))
            for x in df_start["Gaze point X (MCSnorm)"].values
        ],
        "y": [
            float(y.replace(",", "."))
            for y in df_start["Gaze point Y (MCSnorm)"].values
        ],
    }
    animator = GazeAnimator(data, duration)
    animation = animator.animate()
