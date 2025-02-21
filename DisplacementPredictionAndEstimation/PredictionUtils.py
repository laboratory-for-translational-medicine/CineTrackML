# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import convolve

def augment(signal, transition_length = 5):
    """
    Augment initial training data, by adding a bit of variation in it.
    Scale the training data and return the scaler for the upcoming measurement to be made.
    output:
        displacement = the signal 1% decreased magnitude  
                        + smoothed signal 5% increased magnitude  
                        + the signal 1% increased magnitude
                        + smoothed signal 5% decreased magnitude
                        + the signal 
        scaler
    """
    # Calculate differences to find the best starting and end points 
    signal = signal.values.flatten() 
    # Apply smoothing using a simple moving average (kernel of size 5)
    smoothing_kernel = np.ones(5) / 5
    smoothed_signal = convolve(signal, smoothing_kernel, mode='same')
    # Helper function to create smooth transitions
    def smooth_transition(start_signal, end_signal):
        """Blend two signals over the transition length."""
        weights = np.linspace(0, 1, transition_length)
        blended_start = start_signal[-transition_length:] * (1 - weights)
        blended_end = end_signal[:transition_length] * weights
        return np.concatenate([start_signal[:-transition_length], blended_start + blended_end, end_signal[transition_length:]])

    # Combine signals with smooth transitions
    augmented_signal = signal*0.99
    augmented_signal = smooth_transition(augmented_signal, smoothed_signal*1.05)
    augmented_signal = smooth_transition(augmented_signal, signal*1.01)
    augmented_signal = smooth_transition(augmented_signal, smoothed_signal*0.95)
    augmented_signal = smooth_transition(augmented_signal, signal)

    return augmented_signal.reshape(-1, 1)

def augmentAndScale(signal, transition_length = 5):
    """
    Augment initial training data, by adding a bit of variation in it.
    Scale the training data and return the scaler for the upcoming measurement to be made.
    output:
        displacement = the signal 1% decreased magnitude  
                        + smoothed signal 5% increased magnitude  
                        + the signal 1% increased magnitude
                        + smoothed signal 5% decreased magnitude
                        + the signal 
        scaler
    """
    # Calculate differences to find the best starting and end points 
    signal = signal.values.flatten() 
    # Apply smoothing using a simple moving average (kernel of size 5)
    smoothing_kernel = np.ones(5) / 5
    smoothed_signal = convolve(signal, smoothing_kernel, mode='same')
    # Helper function to create smooth transitions
    def smooth_transition(start_signal, end_signal):
        """Blend two signals over the transition length."""
        weights = np.linspace(0, 1, transition_length)
        blended_start = start_signal[-transition_length:] * (1 - weights)
        blended_end = end_signal[:transition_length] * weights
        return np.concatenate([start_signal[:-transition_length], blended_start + blended_end, end_signal[transition_length:]])

    # Combine signals with smooth transitions
    augmented_signal = signal*0.99
    augmented_signal = smooth_transition(augmented_signal, smoothed_signal*1.05)
    augmented_signal = smooth_transition(augmented_signal, signal*1.01)
    augmented_signal = smooth_transition(augmented_signal, smoothed_signal*0.95)
    augmented_signal = smooth_transition(augmented_signal, signal)
    
    # Normalize signal
    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit and transform the signal
    normalized_signal = scaler.fit_transform(augmented_signal.reshape(-1, 1))
    # Return the normalized signal and the scaler
    return normalized_signal, scaler