import torch
import numpy as np

"""
The Audio Intensity Map is generated during the saving stage of a trained 3D Gaussian Splatting scene.
Refer to the `save_ply` method in:
./src/avs_3d/echosegnet/audio_informed_gaussian_splatting/scene/gaussian_model.py
"""

class AudioMapBuilder():
    def __init__(
        self,
        scene_audio, 
        receiver_positions,
        receiver_rotations, 
        points,
        moving_window: int = 6
    ):

        self.scene_audio = scene_audio

        self.receiver_positions = np.array(receiver_positions)
        self.receiver_rotations = np.array(receiver_rotations)

        assert len(self.scene_audio) == len(self.receiver_positions) == len(self.receiver_rotations)

        self.points = np.array(points)
        self.moving_window = moving_window 

        self.num_steps = len(self.receiver_positions)

    @staticmethod
    def compute_rms(tensor):
        """
        Compute the RMS intensity of a given tensor.
        
        Args:
        tensor (torch.Tensor): The audio samples.
        
        Returns:
        float: The RMS intensity.
        """

        return torch.sqrt(torch.mean(tensor ** 2)).item()

    @staticmethod
    def moving_average(data, window_size):
        """
        Apply a moving average filter to the data.
        
        Args:
        data (np.ndarray): The input data to filter.
        window_size (int): The size of the moving window.
        
        Returns:
        np.ndarray: The filtered data with initial values unfiltered.
        """
        filtered_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        initial_values = data[:window_size - 1]  # Keep the initial values unfiltered

        return np.concatenate((initial_values, filtered_data))

    def compute_rms_parallel(self, segments):
        """
        Compute RMS for a list of segments in parallel using multiprocessing.
        
        Args:
        segments (list of torch.Tensor): The segments to compute RMS for.
        
        Returns:
        list of float: The computed RMS values.
        """

        rms_values = []
        
        for segment in segments:
            rms_values.append(
                self.compute_rms(segment)
            )

        return rms_values
    
    def indicator_function_vectorized(self, louder_side, receiver_positions, receiver_rotations, points):
        """
        Vectorized version of the indicator function.
        
        Args:
        louder_side (str): The louder side ('l' or 'r').
        receiver_positions (np.ndarray): Array of receiver positions.
        receiver_rotations (np.ndarray): Array of receiver rotations.
        points (np.ndarray): Array of points.
        
        Returns:
        np.ndarray: Indicator values.
        """
        theta = np.radians(receiver_rotations)
        d = np.stack([np.cos(theta), np.sin(theta)], axis=-1)

        r = receiver_positions[:, np.newaxis, :]
        o = points[np.newaxis, :, :]
        v = o - r

        det = d[:, np.newaxis, 0] * v[:, :, 1] - d[:, np.newaxis, 1] * v[:, :, 0]

        return det > 0 if louder_side == 'l' else det < 0

    def compute_confidence_coefficients(self):
        """
        Compute the confidence coefficients for given left and right channel audio data.
        """

        EPSILON = 1e-6
        
        louder_sides = []
        confidence_coefficients = np.zeros(len(self.points))
        indicator_values = np.zeros((self.num_steps, len(self.points)))

        segments_left = [self.scene_audio[k][0] for k in self.scene_audio.keys()]
        segments_right = [self.scene_audio[k][1] for k in self.scene_audio.keys()]

        # Compute RMS values 
        rms_left_history = self.compute_rms_parallel(segments_left)
        rms_right_history = self.compute_rms_parallel(segments_right)

        # Apply moving average
        rms_left_history = self.moving_average(np.array(rms_left_history), self.moving_window)
        rms_right_history = self.moving_average(np.array(rms_right_history), self.moving_window)

        for t in range(self.num_steps):

            louder_side = 'l' if rms_left_history[t] > rms_right_history[t] else 'r'
            louder_sides.append(louder_side)

            WEIGHT = (
                abs(rms_left_history[t] - rms_right_history[t]) 
                / (max(rms_left_history[t], rms_right_history[t]) + EPSILON)
            )

            indicator_values[t] = self.indicator_function_vectorized(louder_side, self.receiver_positions[t:t+1], self.receiver_rotations[t:t+1], self.points).flatten()
            confidence_coefficients += WEIGHT * indicator_values[t]

        confidence_coefficients = confidence_coefficients / np.max(confidence_coefficients)

        return confidence_coefficients, louder_sides, rms_left_history, rms_right_history, indicator_values
    