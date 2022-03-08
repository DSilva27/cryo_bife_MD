
"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import seaborn as sns

#img = np.loadtxt("data/num_images_grid_3_well")
paths = np.load("paths.npy")
images = np.loadtxt("example_data/images.txt")
#images = images - 10*np.ones(images.shape)

min_x = np.min(paths[:,:,0]); min_y = np.min(paths[:,:,1])
max_x = np.max(paths[:,:,0]); max_y = np.max(paths[:,:,1])

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [], lw=2, ls="--", marker="o")

df = pd.DataFrame()

df["x"] = images[:,0]
df["y"] = images[:,1]

#sns.kdeplot(data=df, x="x", y="y", ax=ax, shade=True, cbar=True)

ax.set_xlabel("x1")
ax.set_ylabel("x2")

ax.set_xlim(min_x-1, max_x+1)
ax.set_ylim(min_y-1, max_y+1)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# # animation function.  This is called sequentially
def animate(i, paths):

    line.set_data(paths[i][:,0], paths[i][:,1])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=paths.shape[0], interval=100, blit=True, fargs=(paths,), repeat=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

writergif = animation.PillowWriter(fps=30) 
anim.save('basic_animation.gif', writer=writergif)

plt.savefig("final_step.png", dpi=300)

plt.show()
