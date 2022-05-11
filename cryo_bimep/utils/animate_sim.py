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

def animate_simulation(traj, initial_path, images=None, ref_path=None, anim_file=None):

    min_x = np.min(traj[:,:,0]); min_y = np.min(traj[:,:,1])
    max_x = np.max(traj[:,:,0]); max_y = np.max(traj[:,:,1])

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes()
    line, = ax.plot([], [], lw=2, ls="--", marker="o")

    if images is not None:

        df = pd.DataFrame()

        df["x"] = images[:,0]
        df["y"] = images[:,1]
        sns.kdeplot(data=df, x="x", y="y", ax=ax, shade=True, cbar=True, cmap="vlag")
    
    if ref_path is not None:
        ax.plot(ref_path[:,0], ref_path[:,1], color="black", ls="-", marker="o", label="ground-truth")

    ax.plot(initial_path[:,0], initial_path[:,1], color="orange", ls="-", marker="o", label="initial path")
    ax.legend()

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.set_xlim(min_x-1, max_x+1)
    ax.set_ylim(min_y-1, max_y+1)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # # animation function.  This is called sequentially
    def animate(i, traj):

        line.set_data(traj[i][:,0], traj[i][:,1])
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=traj.shape[0], interval=100, blit=True, fargs=(traj,), repeat=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html

    writergif = animation.PillowWriter(fps=15) 

    if anim_file is not None:
        anim.save(f'{anim_file}.gif', writer=writergif)
        plt.savefig(f"{anim_file}.png", dpi=300)

    plt.show()
