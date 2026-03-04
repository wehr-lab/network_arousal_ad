import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def makeSymmetric(G):
    """
    Make a square matrix symmetric by averaging it with its transpose.
    
    Parameters:
    G (numpy.ndarray): The input square matrix.
    
    Returns:
    numpy.ndarray: The symmetric version of the input matrix.
    """
    return (G + G.T) / 2

def makeBinary(G, threshold=0.5):
    """
    Convert a matrix to binary based on a threshold.
    
    Parameters:
    G (numpy.ndarray): The input matrix.
    threshold (float): The threshold value for binarization.
    
    Returns:
    numpy.ndarray: The binary version of the input matrix.
    """
    return (G >= threshold).astype(int)


def safeRichClub(G, k=None):
    """
    Wrapper for the rich-club coefficient calculation in NetworkX to return only the RC value for a specific k.
    """ 
    

def richClubDegreeN(G, k):
    pass 

def matDensity(mat):
    """Calculate the connection density of a 3D connectivity matrix."""
    total_connections = mat.size
    non_zeros = np.count_nonzero(mat)
    density = non_zeros / total_connections
    return density

def create_sliding_window_graphs(mat_3d, window_size, step_size=None):
    """
    Create sliding window dynamic graphs from 3D Granger causality matrix.
    
    Parameters:
    -----------
    mat_3d : np.ndarray
        3D array of shape (time, n_cells, n_cells) - Granger causality matrices
    window_size : int
        Size of the sliding window in time steps
    step_size : int, optional
        Step size for sliding window. If None, defaults to window_size//2 (50% overlap)
        
    Returns:
    --------
    dynamic_graphs : np.ndarray
        3D array of shape (n_windows, n_cells, n_cells) - max projected graphs
    window_info : dict
        Information about windows including start/end times
    """
    
    if step_size is None:
        step_size = window_size // 2  # 50% overlap by default
    
    n_time, n_cells, _ = mat_3d.shape
    
    # Calculate number of windows
    n_windows = (n_time - window_size) // step_size + 1
    
    # Initialize output array
    dynamic_graphs = np.zeros((n_windows, n_cells, n_cells))
    
    # Track window information
    window_starts = []
    window_ends = []
    
    print(f"Creating {n_windows} windows:")
    print(f"- Window size: {window_size} time steps")
    print(f"- Step size: {step_size} time steps")
    print(f"- Overlap: {window_size - step_size} time steps")
    
    # Create sliding windows
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Extract window data
        window_data = mat_3d[start_idx:end_idx, :, :]
        
        # Max projection within window
        max_proj_window = np.nanmax(window_data, axis=0)
        
        # Store result
        dynamic_graphs[i, :, :] = max_proj_window
        
        # Track window boundaries
        window_starts.append(start_idx)
        window_ends.append(end_idx)
        
        # if i < 5 or i % 50 == 0:  # Print first few and every 50th window
        #     n_edges = np.count_nonzero(max_proj_window)
        #     max_val = np.nanmax(max_proj_window)
        #     print(f"Window {i:3d}: t={start_idx:5d}-{end_idx:5d}, edges={n_edges:4d}, max_GC={max_val:.4f}")
    
    window_info = {
        'window_size': window_size,
        'step_size': step_size,
        'n_windows': n_windows,
        'window_starts': window_starts,
        'window_ends': window_ends,
        'overlap': window_size - step_size
    }
    
    return dynamic_graphs, window_info


def analyze_dynamic_graphs(dynamic_graphs, window_info, title="Dynamic Network Analysis"):
    """
    Analyze temporal patterns in the dynamic graphs. Pass 1 3D array. 
    """
    
    print(f"\n=== {title.upper()} ===")
    
    # Calculate connectivity metrics for each window
    n_windows = dynamic_graphs.shape[0]
    edges_per_window = []
    density_per_window = []
    max_strength_per_window = []
    mean_strength_per_window = []
    
    n_nodes = dynamic_graphs.shape[1]
    max_possible_edges = n_nodes * (n_nodes - 1)  # directed graph
    
    for i in range(n_windows):
        graph = dynamic_graphs[i]
        
        # Count edges (non-zero connections)
        n_edges = np.count_nonzero(graph)
        edges_per_window.append(n_edges)
        
        # Calculate density
        density = n_edges / max_possible_edges
        density_per_window.append(density)
        
        # Strength statistics
        max_strength = np.nanmax(graph)
        mean_strength = np.nanmean(graph[graph > 0]) if n_edges > 0 else 0
        max_strength_per_window.append(max_strength)
        mean_strength_per_window.append(mean_strength)
    
    # Summary statistics
    print(f"Dynamic Graph Statistics:")
    print(f"- Number of windows: {n_windows}")
    print(f"- Window size: {window_info['window_size']} time steps")
    print(f"- Window overlap: {window_info['overlap']} time steps")
    print(f"- Average edges per window: {np.mean(edges_per_window):.1f} ± {np.std(edges_per_window):.1f}")
    print(f"- Average density per window: {np.mean(density_per_window):.4f} ± {np.std(density_per_window):.4f}")
    print(f"- Connection strength range: {np.min(max_strength_per_window):.4f} to {np.max(max_strength_per_window):.4f}") ## this should ideally just be 1
    
    return {
        'edges_per_window': edges_per_window,
        'density_per_window': density_per_window,
        'max_strength_per_window': max_strength_per_window,
        'mean_strength_per_window': mean_strength_per_window
    }

def create_movie_dynamic_graph(mat, save_path=None, fps=20, interval=50, title_prefix="GC MAT"):
    """
    Create an animated movie showing how dynamic graph connectivity changes over time.
    
    Parameters:
    -----------
    mat : np.ndarray
        3D array of shape (time, n_nodes, n_nodes) - connectivity matrices over time
    save_path : str, optional
        Full path where to save the movie (default: "GC_tone_decode.mp4")
    fps : int, optional
        Frames per second for the output video (default: 20)
    interval : int, optional
        Delay between frames in milliseconds for animation (default: 50)
    title_prefix : str, optional
        Prefix for the title shown in each frame (default: "GC MAT")
        
    Returns:
    --------
    None
        Saves the animation as an MP4 file
    """
    
    if save_path is None:
        save_path = "GC_tone_decode.mp4"
    
    # Create figure and initial plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize with first time point
    im = ax.imshow(mat[0], vmin=mat.min(), vmax=mat.max(), cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Connection Strength')
    
    # Set labels and initial title
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    title = ax.set_title(f"{title_prefix} t = 0")
    
    def update(frame):
        """Update function for animation"""
        im.set_data(mat[frame])
        title.set_text(f"{title_prefix} t = {frame}")
        return [im, title]

    # Create animation
    print(f"Creating animation with {mat.shape[0]} frames...")
    ani = FuncAnimation(
        fig, 
        update, 
        frames=mat.shape[0], 
        blit=True, 
        interval=interval,
        repeat=False
    )

    # Save animation
    print(f"Saving animation to: {save_path}")
    ani.save(
        save_path,
        writer="ffmpeg",
        fps=fps,
        dpi=100,
        codec="libx264",
        bitrate=2000,
        extra_args=["-pix_fmt", "yuv420p"],  # broad player compatibility
    )
    
    plt.close(fig)
    print(f"Animation saved successfully to {save_path}")