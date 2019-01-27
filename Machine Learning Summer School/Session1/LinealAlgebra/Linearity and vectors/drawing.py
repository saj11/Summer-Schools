#%matplotlib inline

"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc
rc('animation', html='html5')

from mpl_toolkits.mplot3d import axes3d

from IPython.display import HTML
 
def toXYZ(vectors):
    # transform vectors to X, Y, Z form
    vectors = np.array(vectors).T
    #X, Y, Z = vectors
    
    origins = np.zeros(shape=vectors.shape)
    
    return vectors, origins # X, Y, Z


def draw_vectors_2d(vectors, origins, fig):
    X, Y = origins
    U, V = vectors
    
    ax = fig.add_subplot(111)
    eps = 0.1
    ax.set_xlim([U.min() - eps, U.max() + eps])
    ax.set_ylim([V.min() - eps, V.max() + eps])
    
    ax.set_title("2D vectors",fontsize=14)
    ax.set_xlabel("x",fontsize=12)
    ax.set_ylabel("y",fontsize=12)
    ax.grid(True,linestyle='--',color='0.75')
    
    ax.quiver(X, Y, U, V, color='rrggbbkkccmmyy', angles='xy', scale_units='xy', scale=1)
 
    plt.show()
    
def draw_vectors_3d(vectors, origins, fig):
    # First set up the figure, the axis, and the plot element we want to animate
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        #x = np.linspace(0, 2, 1000)
        #y = np.sin(2 * np.pi * (x - 0.01 * i))
        #line.set_data(x, y)
        angle = 2
        ax.view_init(elev=None, azim=i*angle)

        return line,

    a1 = [1, 0, 0]
    b1 = [0, 1, 0]
    c1 = [0, 0, 1]
    d1 = np.sum([a1, b1, c1], axis=1)

    # transform vectors to X, Y, Z form
    #vectors = np.array([a1, b1, c1, d1]).T
    #print(vectors)
    U, W, V = vectors
    #origins = np.zeros(shape=vectors.shape)
    X, Y, Z = origins

    #ax.plot(X, Y, Z)
    ax.quiver(X, Y, Z, U, V, W, color='rgbkcmy') # up to 7 different vector colors
    
    eps = 0.1
    ax.set_xlim([U.min() - eps, U.max() + eps])
    ax.set_ylim([V.min() - eps, V.max() + eps])
    ax.set_zlim([V.min() - eps, V.max() + eps])

    ax.set_title("3D vectors",fontsize=14)
    ax.set_xlabel("x",fontsize=12)
    ax.set_ylabel("y",fontsize=12)
    ax.set_zlabel("z",fontsize=12)
    ax.grid(True,linestyle='--',color='0.75')

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=180, interval=100, repeat=True, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    #plt.show()

    plt.close()
    return HTML(anim.to_html5_video())