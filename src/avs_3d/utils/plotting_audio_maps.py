import numpy as np
import os
from matplotlib import pyplot as plt

def plot_audio_maps(audio_intensities, grid_points, indicator_values, receiver_positions, receiver_rotations, path: str) -> None:

    plt.figure(figsize=(10, 10))

    scatter = plt.scatter(
        np.array(grid_points)[:, 0], 
        np.array(grid_points)[:, 1], 
        c=np.array(audio_intensities), 
        cmap='viridis', 
        s=3,  
        edgecolors='none'  
    )

    plt.colorbar(scatter, label='Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of Discrete Points with Mask Values')
    plt.axis('equal') 
    plt.tight_layout()

    scatter = plt.scatter(
        np.array(receiver_positions)[:, 0], 
        np.array(receiver_positions)[:, 1],
        cmap='viridis', 
        s=3,  
        edgecolors='none',  
        label='Receiver position'
    )
    
    plt.legend()
    plt.savefig(os.path.join(path, 'audio_intensity_map.png')) 
    plt.close()

    # for FRAME_INDEX in range(len(indicator_values)):

    #     plt.figure(figsize=(10, 10))

    #     scatter = plt.scatter(
    #         np.array(grid_points)[:, 0], 
    #         np.array(grid_points)[:, 1], 
    #         c=np.array(indicator_values[FRAME_INDEX]), 
    #         cmap='viridis', 
    #         s=3,  
    #         edgecolors='none'  
    #     )

    #     plt.colorbar(scatter, label='Intensity')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title(f'Values of indicator function at frame {FRAME_INDEX}')

    #     scatter = plt.scatter(
    #         np.array(receiver_positions)[:, 0], 
    #         np.array(receiver_positions)[:, 1],
    #         cmap='viridis', 
    #         s=3,  # Adjust 's' for point size
    #         edgecolors='none',  
    #         label='Receiver position'
    #     )

    #     end_x = np.cos(np.deg2rad(receiver_rotations)) * 100
    #     end_y = np.sin(np.deg2rad(receiver_rotations)) * 100

    #     # Plot the lines using quiver
    #     plt.quiver(
    #         receiver_positions[FRAME_INDEX, 0], 
    #         receiver_positions[FRAME_INDEX, 1], 
    #         end_x[FRAME_INDEX], 
    #         end_y[FRAME_INDEX], 
    #         angles='xy', 
    #         scale_units='xy', 
    #         width=0.01,
    #         scale=1,
    #         color='r', 
    #         label='Direction'
    #     )
        
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(path, f'indicator_values_{FRAME_INDEX}.png')) 
    #     plt.close()

    ######################################################################