# Colours

# Imports
import numpy as np
import matplotlib.colors
import colorcet as cc
import pylab as pl
import os

#Generate Colormap Intensity
def colours():
    cmap = np.zeros([256, 4])
    cmap[:, 3] = np.linspace(0, 1, 256)
    cmap[:, 0]= np.linspace(0, 0, 256)
    cmap[:, 1]= np.linspace(0, 0, 256)
    cmap[:, 2]= np.linspace(0, 0, 256)
    #Intensity colours
    imap = matplotlib.colors.ListedColormap(cmap)
    #Phase colours
    pmap= cc.m_CET_C6

    ### Saving Colourbars
    cmap = pmap
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig = pl.figure(figsize=(6, 1), dpi=300) 
    ax = fig.add_axes([0, 0, 1, 1]) 
    cb = pl.colorbar(pl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                    cax=ax, 
                    orientation="horizontal", 
                    ticks=[])
    ax.set_axis_off() 
    os.makedirs("images", exist_ok=True)
    pl.savefig("images/phaseColorbar.png", 
            bbox_inches='tight', 
            pad_inches=0)
    pl.close(fig) 

    cmap = 'viridis'
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig = pl.figure(figsize=(6, 1), dpi=300) 
    ax = fig.add_axes([0, 0, 1, 1]) 
    cb = pl.colorbar(pl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                    cax=ax, 
                    orientation="horizontal", 
                    ticks=[])
    ax.set_axis_off() 
    pl.savefig("images/uniform0-1Colorbar.png", 
            bbox_inches='tight', 
            pad_inches=0)
    pl.close(fig) 

    customColoursBGY = [
        (0.,0.047803,0.4883,1.),
        (0.,0.27531,0.72221,1),
        (0.10786,0.56059,0.38276,1.),
        (0.21196,0.82159,0.099996,1.),
        (1.,0.94606,0.13735,1.)
        ]

    customColoursViridis = [
        (0.267004, 0.004874, 0.329415, 1.      ),
        (0.229739, 0.322361, 0.545706, 1.      ),
        (0.127568, 0.566949, 0.550556, 1.      ),
        (0.369214, 0.788888, 0.382914, 1.      ),
        (0.993248, 0.906157, 0.143936, 1.      )
    ]

    return pmap, imap, customColoursBGY, customColoursViridis