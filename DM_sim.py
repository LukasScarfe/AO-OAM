import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from LightPipes import *
import os

# --- 1. LIGHTPIPES & DM CONFIGURATION ---
wavelength = 0.633 * um
size = 28.0 * mm
N = 180              
pitch_val = 2.5 * mm 
coupling = 0.45      
pupil_diam = 22.5 * mm

# --- 2. CUSTOM COLOURS INTEGRATION ---
try:
    from functions.colours import colours
    pmap, imap, _, _ = colours()
except ImportError:
    pmap, imap = 'hsv', 'gray'

# --- 3. GEOMETRY & INFLUENCE FUNCTIONS ---
x_grid = np.linspace(-size/2, size/2, N)
X, Y = np.meshgrid(x_grid, x_grid)

act_x, act_y = [], []
for i in np.arange(-5, 6):
    for j in np.arange(-5, 6):
        px, py = i * 2.5 * mm, j * 2.5 * mm 
        if np.sqrt(px**2 + py**2) < 13.8 * mm:
            act_x.append(px)
            act_y.append(py)

act_x, act_y = np.array(act_x), np.array(act_y)
num_act = len(act_x)
commands = np.zeros(num_act) # Now storing physical stroke in meters

sigma = (2.5 * mm) / np.sqrt(-2 * np.log(coupling))
influences = np.array([np.exp(-((X - px)**2 + (Y - py)**2) / (2 * sigma**2)) 
                       for px, py in zip(act_x, act_y)])

# --- 4. LIVE PLOTTING UI ---
fig, (ax_phase, ax_psf) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

F_init = Begin(size, wavelength, N)
F_init = CircAperture(F_init, pupil_diam/2)

im_phase = ax_phase.imshow(np.zeros((N, N)), extent=[-size/2/mm, size/2/mm, -size/2/mm, size/2/mm], 
                           cmap=pmap, vmin=0, vmax=2*np.pi, origin='lower')
im_psf = ax_psf.imshow(np.zeros((N, N)), extent=[-size/2/mm, size/2/mm, -size/2/mm, size/2/mm], 
                       cmap=imap, origin='lower')

dots = ax_phase.scatter(act_x/mm, act_y/mm, c='white', s=10, alpha=0.2, picker=True)
selector = ax_phase.scatter([0], [0], s=100, facecolors='none', edgecolors='gold', lw=2)

ax_phase.set_title("Live Phase (-2π to 2π Control)")
ax_psf.set_title("Live Focal Spot")

# --- 5. INTERACTION LOGIC ---
selected_idx = 48

def update(val):
    global commands
    total_surface = np.tensordot(commands, influences, axes=(0, 0))
    # Double pass reflection: Phase = (2*pi/lambda) * 2 * surface
    opt_phase = (4 * np.pi * total_surface) / wavelength
    wrapped_phase = np.mod(opt_phase, 2*np.pi)
    
    F_dm = SubPhase(F_init, opt_phase)
    F_far = Lens(F_dm, 50*cm)
    F_far = Forvard(F_far, 50*cm)
    psf_int = Intensity(F_far)
    
    im_phase.set_data(wrapped_phase)
    im_psf.set_data(np.sqrt(psf_int))
    im_psf.set_clim(0, np.max(np.sqrt(psf_int)) * 0.9)
    fig.canvas.draw_idle()

def on_pick(event):
    global selected_idx
    selected_idx = event.ind[0]
    selector.set_offsets(np.c_[act_x[selected_idx]/mm, act_y[selected_idx]/mm])
    
    # Convert current physical stroke back to phase for the slider
    current_phase = (4 * np.pi * commands[selected_idx]) / wavelength
    stroke_slider.eventson = False
    stroke_slider.set_val(current_phase)
    stroke_slider.eventson = True
    update(None)

def set_oam(event):
    global commands
    angles = np.arctan2(act_y, act_x)
    angles = (angles + np.pi) % (2 * np.pi)
    # For l=1, we need 2pi phase ramp. Max stroke = lambda / 2
    max_stroke = wavelength / 2.0
    commands = (angles / (2 * np.pi)) * max_stroke
    update(None)

def save_csv(event):
    total_surface = np.tensordot(commands, influences, axes=(0, 0))
    opt_phase = (4 * np.pi * total_surface) / wavelength
    np.savetxt("dm_phase_profile.csv", opt_phase, delimiter=",")
    print("Saved 2D phase profile (radians) to dm_phase_profile.csv")

# --- 6. WIDGETS ---
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
# Slider label updated to 'Phase (rad)' to be accurate
stroke_slider = Slider(ax_slider, 'Phase (rad)', -2*np.pi, 2*np.pi, valinit=0.0)

def slider_update(val):
    # Convert slider phase (val) to physical stroke (meters)
    # stroke = (Phase * lambda) / (4 * pi)
    commands[selected_idx] = (val * wavelength) / (4 * np.pi)
    update(None)

stroke_slider.on_changed(slider_update)
fig.canvas.mpl_connect('pick_event', on_pick)

# Buttons
btn_oam = Button(plt.axes([0.5, 0.02, 0.1, 0.04]), 'OAM Mode')
btn_oam.on_clicked(set_oam)

btn_csv = Button(plt.axes([0.61, 0.02, 0.1, 0.04]), 'Save CSV')
btn_csv.on_clicked(save_csv)

btn_flat = Button(plt.axes([0.72, 0.02, 0.1, 0.04]), 'Flatten')
btn_flat.on_clicked(lambda e: [commands.fill(0), stroke_slider.reset(), update(None)])

print(f"DM initialized. Controls mapped to phase radians.")
update(None)
plt.show()