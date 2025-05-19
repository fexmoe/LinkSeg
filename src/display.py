import jams
import librosa
import numpy as np
import matplotlib.pyplot as plt
"""
# Load JAMS file
jam = jams.load("../fur-elise/predictions/fur-elise.jams")


# Load audio file
audio_path = "../fur-elise/audio/fur-elise.wav"  # adjust path as needed
y, sr = librosa.load(audio_path)

# Plot the waveform
plt.figure(figsize=(14, 8))

# Plot the waveform
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
plt.title('Prediction for "Für Elise"')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Extract segment boundaries from JAMS file
segment_data = jam.annotations[0]
boundaries = [obs.time for obs in segment_data.data]
labels = [obs.value for obs in segment_data.data]

# Add vertical lines for segment boundaries
for time in boundaries:
    plt.axvline(x=time, color='r', linestyle='--', alpha=0.7)

# Color the waveform based on segments
plt.subplot(2, 1, 1)
for i in range(len(boundaries)-1):
    # Get the segment start and end indices in the audio array
    start_idx = int(boundaries[i] * sr)
    end_idx = int(boundaries[i+1] * sr)
    # Get the time slice for this segment
    segment_time = np.linspace(
        boundaries[i], boundaries[i+1], end_idx - start_idx)
    # Plot this segment with a specific color
    # Use a darker colormap (tab10) with more distinct colors
    # and modulo to a smaller number to get more contrast between adjacent segments
    color = plt.cm.Dark2(hash(labels[i]) % 8 / 8.0)
    plt.plot(segment_time, y[start_idx:end_idx], color=color)
    # Add label text above the segment
    plt.text((boundaries[i] + boundaries[i+1])/2, max(y[start_idx:end_idx])*1.1, labels[i],
             horizontalalignment='center', color='black')

# Handle the last segment
if len(labels) >= len(boundaries):
    start_idx = int(boundaries[-1] * sr)
    end_idx = len(y)  # Use the end of the audio
    segment_time = np.linspace(boundaries[-1], len(y)/sr, end_idx - start_idx)
    color = plt.cm.Dark2(hash(labels[-1]) % 8 / 8.0)
    plt.plot(segment_time, y[start_idx:end_idx], color=color)
    plt.text(boundaries[-1] + (len(y)/sr - boundaries[-1])/2, max(y[start_idx:end_idx])*1.1, labels[-1],
             horizontalalignment='center', color='black')

plt.show()

# If you want to also hear the audio (optional)
# import IPython.display as ipd
# ipd.Audio(y, rate=sr)
"""


def display_self_similarity_matrix(npy_path, cmap='viridis'):
    """
    Display a self-similarity matrix stored as a numpy tensor (.npy file)

    Parameters:
    -----------
    npy_path : str
        Path to the .npy file containing the self-similarity matrix
    cmap : str, optional
        Colormap for visualization (default: 'viridis')
    """
    # Load the self-similarity matrix
    ssm = np.load(npy_path)

    # Increase font sizes
    plt.rcParams.update({
        'font.size': 42,
        'axes.titlesize': 24,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })

    # Create a new figure
    plt.figure(figsize=(10, 10))

    # Display the matrix as an image
    plt.imshow(ssm, origin='lower', aspect='equal', cmap=cmap)

    plt.colorbar(label='Similarity')
    plt.clim(np.min(ssm), np.max(ssm))  # Set color limits to data range
    plt.title('Self-Similarity Matrix')
    plt.xlabel('Time (frames)')
    plt.ylabel('Time (frames)')

    plt.tight_layout()

    # Save the figure as an image with increased font size
    plt.savefig(npy_path.replace('.npy', '_ssm.png'),
                dpi=300, bbox_inches='tight')

    # Display the figure
    plt.show()


def display_multiple_ssms(npy_paths, titles=None, cmap='viridis'):
    """
    Display multiple self-similarity matrices side by side

    Parameters:
    -----------
    npy_paths : list
        List of paths to .npy files containing the self-similarity matrices
    titles : list, optional
        List of titles for each matrix (default: None)
    cmap : str, optional
        Colormap for visualization (default: 'viridis')
    """
    n = len(npy_paths)
    if titles is None:
        titles = [f'SSM {i+1}' for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=(10*n, 10))
    if n == 1:
        axes = [axes]

    for i, (path, title, ax) in enumerate(zip(npy_paths, titles, axes)):
        ssm = np.load(path)
        im = ax.imshow(ssm, origin='lower', aspect='equal', cmap=cmap)
        fig.colorbar(im, ax=ax, label='Similarity')
        ax.set_title(title)
        ax.set_xlabel('Time (frames)')
        if i == 0:  # Only add y-label for the first plot
            ax.set_ylabel('Time (frames)')

    plt.tight_layout()
    plt.show()
    return fig, axes


# display_multiple_ssms(["tensor_debug/A.npy", "tensor_debug/A-2.npy"],
#                      titles=["A", "A'"])

# Example usage:
display_self_similarity_matrix("tensor_debug/A-2.npy")
