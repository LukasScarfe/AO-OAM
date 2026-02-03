import os, subprocess

import numpy as np
import matplotlib.pyplot as plt

from LightPipes import Field, Intensity, Phase


from functions.colours import colours
pmap, imap, _, _ = colours()


#Show a plot of a single beam, or an array of beams with phase and intensity
def plot_beam(Fs: list[Field],rows: int=1,aperature: float=0,intensity: bool=True,phase: bool=True,transparent: bool=False, dpi: int=300) -> plt.Figure:
    """
    Create and return a matplotlib plot of an LightPipes beam or a list of them, in a row. 
    
    :param Fs: Single LightPipe beam or a list of LightPipe beams to plot
    :type Fs: list[Field] or Field
    :param rows: number of rows to plot if plotting many beams
    :type rows: int
    :param aperature: size of aperture in m to show on each beam.
    :type aperature: float
    :param intensity: Show the intensity of the beam with greyscale?
    :type intensity: bool
    :param phase: show the phase of the beam with a cyclical colour?
    :type phase: bool
    :param dpi: dpi to create the figure at
    :type dpi: int
    :return: Resultant matplotlib figure
    :rtype: Figure
    """
    
    if not hasattr(Fs, "__len__"):
        Fs=[Fs]

    totalModes = len(Fs)
    if rows>=totalModes:
        columns=1
        rows=totalModes
    else:
        columns = totalModes//rows + (1 if totalModes%rows else 0)
    Position = range(1,totalModes + 1)

    fig_width = columns
    fig_height = rows

    fig = plt.figure(1,figsize=(fig_width, fig_height),dpi=dpi)

    pixels,size=Fs[0].N,Fs[0].siz

    for index,F in enumerate(Fs):
        ax = fig.add_subplot(rows,columns,Position[index])
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.margins(0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        Phi=np.mod(Phase(F),2*np.pi)

        if transparent==True:
            I = Intensity(F)
            norm_I = (I - I.min()) / (I.max() - I.min()) if I.max() > I.min() else I
            rgba = pmap(Phi / (2 * np.pi))
            rgba[..., 3] = norm_I if intensity else 1.0
            
            if not phase: 
                rgba[..., :3] = 0.5
                
            ax.imshow(rgba, interpolation='nearest', aspect='auto')
        else: 
        
            I=1-Intensity(1,F)
            
            ax.set_facecolor('black')

            ax.imshow(Phi,cmap=pmap,vmin=0,vmax=2*np.pi,interpolation='None') if phase==True else None
            ax.imshow(I,cmap=imap if phase==True else plt.colormaps['gray_r'] ,vmin=np.min(I),vmax=np.max(I),interpolation='None') if intensity ==True else None

            if aperature:
                centre=(pixels/2-0.5,pixels/2-0.5) if pixels%2 == 1 else (pixels/2,pixels/2)
                circle = plt.Circle(centre,aperature*pixels/size, color='w', fill=False,linewidth=0.5)
                ax.add_patch(circle)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)

    return fig

# This should be depreciated now that it has been integrated into the the default plot beam?
# def plotBeamTransparent(Fs, rows=1, intensity=True, phase=True, dpi=300):
#     Fs = Fs if isinstance(Fs, list) else [Fs]
#     num_beams = len(Fs)
#     cols = (num_beams + rows - 1) // rows
    
#     # Use the grid size of the field (e.g., 512)
#     N = Fs[0].N 
    
#     # Calculate figure size in inches to hit exact pixel targets
#     fig_w = (cols * N) / dpi
#     fig_h = (rows * N) / dpi
    
#     # Create figure with NO frame or padding
#     fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='none')
    
#     for i in range(num_beams):
#         # Calculate position for each subplot manually to ensure 0 padding
#         # [left, bottom, width, height] in normalized coordinates (0 to 1)
#         col_idx = i % cols
#         row_idx = rows - 1 - (i // cols) # Start from top
        
#         ax_pos = [col_idx/cols, row_idx/rows, 1/cols, 1/rows]
#         ax = fig.add_axes(ax_pos)
#         ax.set_axis_off()
        
#         F = Fs[i]
#         I = Intensity(F)
#         P = np.mod(Phase(F), 2 * np.pi)
        
        
#         norm_I = (I - I.min()) / (I.max() - I.min()) if I.max() > I.min() else I
#         rgba = pmap(P / (2 * np.pi))
#         rgba[..., 3] = norm_I if intensity else 1.0
        
#         if not phase: 
#             rgba[..., :3] = 0.5
            
#         ax.imshow(rgba, interpolation='nearest', aspect='auto')

#     return fig


def create__web_movie(image_folder: str, output_location: str, fps: int=60) -> None:
    """
    Create movie in webm format so that it can use transparency. Takes all the images from the image_folder and uses ffmpeg.
    
    :param image_folder: Folder with all images to become a movie
    :type image_folder: str
    :param output_location: output file location to place the movie
    :type output_location: str
    :param fps: Frames per Second
    :type fps: int
    """
    # Ensure filenames are sorted (e.g., output0.png, output1.png...)
    # We use a text file list for FFmpeg to handle non-sequential naming
    input_list = "images.txt"
    filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    with open(input_list, "w") as f:
        for fname in filenames:
            f.write(f"file '{os.path.join(image_folder, fname)}'\n")

    # FFmpeg command for VP9 WebM with Alpha:
    # -f concat: use the text file list
    # -pix_fmt yuva420p: The 'a' stands for Alpha (transparency support)
    # -auto-alt-ref 0: Required for transparency in some VP9 versions
    cmd = [
        'ffmpeg', '-y', 
        '-r', str(fps), 
        '-f', 'concat', '-safe', '0', '-i', input_list,
        '-c:v', 'libvpx-vp9', 
        '-pix_fmt', 'yuva420p', 
        '-auto-alt-ref', '0', 
        output_location
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully saved transparent video: {output_location}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
    finally:
        if os.path.exists(input_list):
            os.remove(input_list)


def plotCrosstalk(crosstalk_matrix: list[float]) -> plt.Figure:
    """
    Does a plot of a crosstalk matrix
    
    :param crosstalk_matrix: crosstalk matrix to be plotted
    :type crosstalk_matrix: list[float]
    :return: matplotlib plot of crosstalk matrix
    :rtype: Figure
    """
    fig = plt.figure(1)
    plt.axis('off')
    plt.imshow(crosstalk_matrix, interpolation='none', cmap='viridis', vmin=0,vmax=1)
    return fig