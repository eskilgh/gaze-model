import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

df = pd.read_csv('export.tsv', sep='\t')


clean = df.dropna(subset=['Gaze point X', 'Gaze point Y'])

image = Image.open('random.png')
im = plt.imread("random.png")

fig, ax = plt.subplots(figsize=(8, 7))
ax.set_xlim(0, image.width)
ax.set_ylim(0, image.height)
ax.imshow(im, extent=[0, image.width, 0, image.height], zorder=0, alpha=0.3)
line_plot, = ax.plot(0, 0)

x_data = [ cols['Gaze point X'] for i, cols in clean.head(4).iterrows()]
y_data = [ cols['Gaze point Y'] for i, cols in clean.head(4).iterrows()]

def animation_frame(row):
    x_data.pop(0)
    y_data.pop(0)
    x_data.append(row[1]['Gaze point X'])
    y_data.append(row[1]['Gaze point Y'])
    line_plot.set_xdata(x_data)
    line_plot.set_ydata(y_data)

    return line_plot


animation = FuncAnimation(fig, func=animation_frame, frames=clean.tail(-4).iterrows(), interval=30, repeat=False)

animation.save('gaze_animation.avi')
plt.show()

