import numpy as np


from setup import BS_LOCATIONS, FIELD_HEIGHT, FIELD_LENGTH

def visualize_network(ax, ue_loc):
    assert np.shape(ue_loc) == (3,)
    ax.set_xlim(left=0,right=FIELD_LENGTH)
    ax.set_ylim(bottom=0,top=FIELD_LENGTH)
    ax.set_zlim(bottom=0,top=FIELD_HEIGHT)
    # plot basestations
    ax.scatter3D(xs=BS_LOCATIONS[:,0], ys=BS_LOCATIONS[:,1], zs=BS_LOCATIONS[:,2], marker="1", s=100)
    # plot user equipment
    ax.scatter3D(xs=ue_loc[0], ys=ue_loc[1], zs=ue_loc[2], marker="*", s=50)
    return