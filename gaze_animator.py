from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np


plt.rcParams[
    "animation.ffmpeg_path"
] = "/Users/eskilgaarehostad/miniconda3/envs/data/bin/ffmpeg"


class GazeAnimator:
    def __init__(self, data, duration, img=None, fps: int = 30):
        assert len(data["time"]) > 0 and (
            len(data["time"]) == len(data["x"]) == len(data["y"]) > 0
        )
        assert fps <= 30, "Does not support fps above 30."

        self.time = data["time"]
        self.x = data["x"]
        self.y = data["y"]
        self.duration = duration
        self.img = img
        self.fps = fps
        self.interval = 1000 // fps
        self.lines = []

    def split_data(self):
        nr_of_frames = int(np.rint((self.duration * self.fps) / 10 ** 3))
        return (
            np.array_split(self.time, nr_of_frames),
            np.array_split(self.x, nr_of_frames),
            np.array_split(self.y, nr_of_frames),
        )

    def animate(self, title, show=False):
        fig = plt.figure(figsize=(8, 7), facecolor="black")

        ax1 = plt.axes(xlim=(0, 1), ylim=(0, 1))
        # Remove ticks from axes
        plt.xticks([], [])
        plt.yticks([], [])
        if title:
            title_obj = plt.title(title)
            plt.setp(title_obj, color="r")

        # origo is located top left
        ax1.invert_yaxis()

        self.line = ax1.plot([], [], lw=1, marker=".", color="red")[0]
        self.point = ax1.plot(
            [],
            [],
            marker="o",
            color="white",
            markeredgecolor="red",
            ls="",
        )[0]
        time, x, y = self.split_data()

        animation = FuncAnimation(
            fig,
            self.anim_frame,
            init_func=self.anim_init,
            frames=zip(x, y),
            interval=self.interval,
            repeat=False,
            blit=True,
            save_count=len(x),
        )
        if self.img:
            im = plt.imread(self.img)
            ax1.imshow(im, extent=[0, 1, 0, 1], zorder=0, alpha=0.8)

        if show:
            plt.show()
        return animation

    def anim_init(self):
        self.line.set_data([], [])
        self.point.set_data([], [])
        return [self.line, self.point]

    def anim_frame(self, points):
        x_vals, y_vals = points
        self.line.set_data(x_vals, y_vals)
        self.point.set_data(x_vals[-1], y_vals[-1])
        return [self.line, self.point]


if __name__ == "__main__":
    import pandas as pd

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
        (df["Presented Media name"] == "test1.png") & (df["Participant name"] == "Siri")
    ]
    df_start = df.tail(1200).dropna()
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
    animator = GazeAnimator(data, duration, img="test1.png")
    anim = animator.animate("Eye gaze trajectory, Participant B", show=False)
    writer = animation.FFMpegWriter(fps=30)
    anim.save("animation.mp4", writer=writer)
