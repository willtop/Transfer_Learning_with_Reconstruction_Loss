import numpy as np
from setup import *

def visualize_network(ax, ue_loc):
    assert np.shape(ue_loc) == (3,)
    ax.set_xlim(left=0,right=FIELD_LENGTH)
    ax.set_ylim(bottom=0,top=FIELD_LENGTH)
    ax.set_zlim(bottom=0,top=FIELD_HEIGHT)
    # plot basestations
    ax.scatter3D(xs=BS_LOCATIONS[:,0], ys=BS_LOCATIONS[:,1], zs=BS_LOCATIONS[:,2], marker="1", s=300, label="BS")
    # plot user equipment
    ax.scatter3D(xs=ue_loc[0], ys=ue_loc[1], zs=ue_loc[2], marker="*", s=75, c='k', label="Ground Truth")
    return

def plot_location_in_network(ax, location, plot_color, plot_label):
    assert np.shape(location) == (3,)
    ax.scatter3D(xs=location[0], ys=location[1], zs=location[2], color=plot_color, marker="o", s=50, label=plot_label)
    return

def generate_circular_gaussians(size_to_generate, per_element_var=1):
    return np.random.normal(size=size_to_generate, loc=0, scale=np.sqrt(per_element_var)) + \
            1j * np.random.normal(size=size_to_generate, loc=0, scale=np.sqrt(per_element_var))