import numpy as np
from setup import *

def visualize_network(ax, ue_loc):
    assert np.shape(ue_loc) == (3,)
    # add in boundaries of the 3D space
    ax.plot([0,FIELD_LENGTH],[0,0],zs=[FIELD_HEIGHT, FIELD_HEIGHT], color='0.4', linewidth=0.8, linestyle='--')
    ax.plot([FIELD_LENGTH,FIELD_LENGTH],[0,FIELD_LENGTH],zs=[FIELD_HEIGHT, FIELD_HEIGHT], color='0.4', linewidth=0.8, linestyle='--')
    ax.plot([FIELD_LENGTH,FIELD_LENGTH],[0,0],zs=[0, FIELD_HEIGHT], color='0.4', linewidth=0.8, linestyle='--')
    # plot basestations
    ax.scatter3D(xs=BS_LOCATIONS[:,0], ys=BS_LOCATIONS[:,1], zs=BS_LOCATIONS[:,2], marker="1", s=1000)
    # label basestations in the plot (better than legend)
    for i in range(N_BS):
        ax.text(x=BS_LOCATIONS[i,0], y=BS_LOCATIONS[i,1], z=BS_LOCATIONS[i,2]+2, s='BS', fontsize=15)
    # plot user equipment
    ax.scatter3D(xs=ue_loc[0], ys=ue_loc[1], zs=ue_loc[2], marker="*", s=70, c='k', label="Ground Truth")
    return

def plot_location_in_network(ax, location, plot_color, plot_label):
    assert np.shape(location) == (3,)
    ax.scatter3D(xs=location[0], ys=location[1], zs=location[2], color=plot_color, marker="X", s=40, label=plot_label)
    return

def bound_3D_region(ax):
    # set the range at the end to ensure the boundaries are enforced
    ax.set_xlim(left=0,right=FIELD_LENGTH)
    ax.set_ylim(bottom=0,top=FIELD_LENGTH)
    ax.set_zlim(bottom=0,top=FIELD_HEIGHT)
    ax.margins(x=0)
    return

def generate_circular_gaussians(size_to_generate, per_element_var=1):
    return np.random.normal(size=size_to_generate, loc=0, scale=np.sqrt(per_element_var)) + \
            1j * np.random.normal(size=size_to_generate, loc=0, scale=np.sqrt(per_element_var))