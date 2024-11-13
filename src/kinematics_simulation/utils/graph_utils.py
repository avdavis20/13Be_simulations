from dataclasses import dataclass
import h5py 
import numpy as np  
import matplotlib.pyplot as plt  
from pathlib import Path 
from typing import List, Optional, Union
from tqdm import tqdm
from .reader_utils import dataReader
from .writer_utils import dataWriter
from matplotlib.figure import Figure


"""
COPY/PASTE THIS ON USER END TO CREATE GRAPHS: 
"""

@dataclass
class GraphSpecs:
    data_profile: str               # What kind of data we are reading in (e.g., "trace", "eloss", etc.)
    graph_dimension: int            # Set 1D, 2D, or 3D graph
    graph_type: str                 # Set 'histogram', 'scatter', '2D_histogram'
    color_scheme: str = ''          # Set 'detected', 'particle', or default to no color-coding
    x_min: Optional[float] = None   # Set x-axis minimum (None for auto-scaling)
    x_max: Optional[float] = None   # Set x-axis maximum (None for auto-scaling)
    y_min: Optional[float] = None   # Set y-axis minimum (None for auto-scaling)
    y_max: Optional[float] = None   # Set y-axis maximum (None for auto-scaling)
    z_min: Optional[float] = None   # Set z-axis minimum (for 3D)
    z_max: Optional[float] = None   # Set z-axis maximum (for 3D)
    log_x: bool = False             # Set log scale for x-axis
    log_y: bool = False             # Set log scale for y-axis
    log_z: bool = False             # Set log scale for z-axis
    point_size: int = 2             # Set size of data point on graph
    bins: int = 100                 # Set number of bins (for histograms)
    title: str = ''                 # Set title of the graph
    x_label: str = ''               # Set label for x-axis
    y_label: str = ''               # Set label for y-axis
    z_label: str = ''               # Set label for z-axis (for 3D graphs)
    gate_detected: bool = False     # Gate on detected particles
    gate_particle: Union[str, List[str]] = ''  # Gate on particle type (e.g., 'target', 'projectile', etc.), can be a list for multiple particles
    

class GraphUtils:
    def __init__(self, file: str, data_profile: str):
        self.file = Path(file)
        self.reader = dataReader(self.file,data_profile)
        self.data_profile = data_profile  # Type of data being loaded ('trace', 'eloss', etc.)
        
        # Initialize general graphing storage
        self.eloss_data: Optional[dict] = None
        self.trace_data: Optional[dict] = None 
        self.data = Optional[dict] = None

################################# PHASE 1: LOAD DATA #################################

    def load_trace_data(self):
        self.data = self.trace_data = self.reader.read_trace_data()
        self.reader.print_all_trace_info()
        if not self.trace_data:
            print("Warning: No trace data found.")
        else:
            print(f"Loaded {len(self.eloss_data)} events, including potentially empty events.")

    def load_eloss_data(self):  
        self.data = self.eloss_data = self.reader.read_eloss_data()
        self.isotopes = self.eloss_data.get('isotopes')
        self.is_detected = self.eloss_data.get('is_detected')

    def load_data(self):
        """
        Loads data based on the specified data profile.
        """
        if self.data_profile == 'trace':
            self.load_trace_data()
        elif self.data_profile == 'eloss':
            self.load_eloss_data()
        else:
            raise ValueError(f"Unsupported data profile: {self.data_profile}")

################################# PHASE 2: PROCESS DATA #################################


    def process_data(self, specs: GraphSpecs):
        """
        Applies gates and color codes based on detected particles and/or particle type.
        Returns gated x and y data, along with corresponding colors and labels.
        """
        
        x_data = np.array(self.data['x'])
        y_data = np.array(self.data['y'])

        # Initialize lists for gated x, y data, colors, and labels
        gated_x = []
        gated_y = []
        colors = []
        labels = []

        # Define masks for different particle types if eloss data is provided
        if self.data_profile == 'eloss':
            target_mask = self.isotopes[:, 0] == '2H'
            projectile_mask = self.isotopes[:, 1] == '12Be'
            ejectile_mask = self.isotopes[:, 2] == '1H'
            residual_mask = self.isotopes[:, 3] == '13Be'

            particle_masks = {
                'target': (target_mask, 'blue', 'Target'),
                'projectile': (projectile_mask, 'green', 'Projectile'),
                'ejectile': (ejectile_mask, 'purple', 'Ejectile'),
                'residual': (residual_mask, 'orange', 'Residual'),
            }

            # If specs.gate_particle is a list, iterate through the particle types
            if isinstance(specs.gate_particle, list):
                for particle in specs.gate_particle:
                    if particle in particle_masks:
                        mask, color, label = particle_masks[particle]
                        if np.any(mask):
                            gated_x.append(x_data[mask])
                            gated_y.append(y_data[mask])
                            colors.append(color)
                            labels.append(label)
            # If it's a single particle type, proceed normally
            elif specs.gate_particle in particle_masks:
                mask, color, label = particle_masks[specs.gate_particle]
                if np.any(mask):
                    gated_x.append(x_data[mask])
                    gated_y.append(y_data[mask])
                    colors.append(color)
                    labels.append(label)

        # If it's trace data, don't gate on or color code anything
        if specs.data_profile == 'trace':
            gated_x = [x_data]
            gated_y = [y_data]
            colors.append('black')
            labels.append('Signal Amplitude')

        # Apply detected gate if specified
        if specs.gate_detected:
            detected_mask = self.is_detected == True
            gated_x = [x[detected_mask] for x in gated_x]
            gated_y = [y[detected_mask] for y in gated_y]

        return gated_x, gated_y, colors, labels
    
    ################################# TRACE DATA PLOTTING METHODS #################################

    def plot_trace_signal(self, event_key: str):
        """
        Plot the signal for a specific event in the trace data.
        Only applicable when the data profile is set to 'trace'.
        """
        if self.data_profile != 'trace':
            print("Error: This function is only applicable to trace data.")
            return

        if event_key not in self.data:
            print(f"Event '{event_key}' not found in loaded data.")
            return

        event_data = self.data[event_key]
        if event_data.shape[0] < 2:
            print(f"Event '{event_key}' does not have data in the expected shape.")
            return
        
        # Extract the data for all available directions
        for i in range(event_data.shape[0]):
            plt.plot(event_data[i, :], label=f"{event_key} - Direction {i + 1}")

        plt.xlabel("Time Bins")
        plt.ylabel("Signal Amplitude")
        plt.title(f"Signal for Event: {event_key}")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"Successfully plotted signals for event '{event_key}'.")

    def plot_all_trace_data(self) -> List[Figure]:
        """
        Plot signals for all events in the trace data.
        Only applicable when the data profile is set to 'trace'.
        
        Returns:
        - figures: List[Figure]. List of matplotlib figures generated for each event.
        """
        if self.data_profile != 'trace':
            print("Error: This function is only applicable to trace data.")
            return []

        print(f"Plotting signals for all {len(self.data)} events...")

        # List to store figures
        figures = []
        skipped_events = []

        for event_key, event_data in tqdm(self.data.items()):
            if event_data.shape[0] < 2:
                skipped_events.append(event_key)
                continue

            # Create a new figure for each event
            fig, ax = plt.subplots(figsize=(10, 6))
            # Extract and plot the data for all available directions
            for i in range(event_data.shape[0]):
                ax.plot(event_data[i, :], label=f"{event_key} - Direction {i + 1}")

            ax.set_xlabel("Time Bins")
            ax.set_ylabel("Signal Amplitude")
            ax.set_title(f"Signal for Event: {event_key}")
            ax.legend()
            ax.grid(True)

            figures.append(fig)

################################# GENERAL PLOTTING METHODS #################################
        
    def plot_graph(self, specs: GraphSpecs):
        """
        Plots a graph based on the given graph specs. Handles 1D, 2D, and 3D graphs.
        """
        if specs.graph_dimension == 1 and specs.graph_type == 'histogram':
            self.plot_1d_histogram(specs)
        elif specs.graph_dimension == 2:
            if specs.graph_type == 'scatter':
                self.plot_2d_scatter(specs)
            elif specs.graph_type == '2D_histogram':
                self.plot_2d_histogram(specs)
        else:
            print(f"Graph dimension '{specs.graph_dimension}' or type '{specs.graph_type}' currently unsupported.")

    def plot_1d_histogram(self, specs: GraphSpecs):
        """
        Plots a 1D histogram based on graph specs.
        """
        gated_x, _, _, _ = self.process_data(specs)
        if gated_x:
            plt.figure(figsize=(12, 8))
            plt.hist(np.concatenate(gated_x), bins=specs.bins, alpha=0.5, color='blue')
            plt.xlabel(specs.x_label)
            plt.ylabel(specs.y_label)
            plt.title(specs.title)
            plt.grid(True)
            plt.show()
        else:
            print("Data is not available for plotting.")
    
    def plot_2d_scatter(self, specs: GraphSpecs):
        """
        Plots a 2D scatter plot based on graph specs.
        """
        gated_x, gated_y, colors, labels = self.process_data(specs)
        if gated_x and gated_y:
            plt.figure(figsize=(12, 8))
            all_x = np.concatenate(gated_x)  # Combine all x data after gating
            all_y = np.concatenate(gated_y)  # Combine all y data after gating
            all_colors = np.concatenate([[colors[i]] * len(gated_x[i]) for i in range(len(gated_x))])  # Repeat each color for the corresponding points

            # Plot all data points in a single scatter call
            plt.scatter(all_x, all_y, c=all_colors, s=specs.point_size)
            plt.xlabel(specs.x_label)
            plt.ylabel(specs.y_label)
            plt.title(specs.title)
            plt.legend(labels)
            plt.grid(True)
            plt.show()
        else:
            print("Data is not available for plotting.")

    def plot_2d_histogram(self, specs: GraphSpecs):
        """
        Plots a 2D histogram based on graph specs.
        """
        gated_x, gated_y, _, _ = self.process_data(specs)
        if gated_x and gated_y:
            plt.figure(figsize=(12, 8))
            plt.hist2d(np.concatenate(gated_x), np.concatenate(gated_y), bins=specs.bins, cmap='viridis')
            plt.colorbar()
            plt.xlabel(specs.x_label)
            plt.ylabel(specs.y_label)

    def cluster_graphs(self, specs_list: List[GraphSpecs]):
        """
        Automatically formats and arranges graphs in a grid using subplot mosaic.
        """
        cluster_num = len(specs_list)
        rows = int(np.sqrt(cluster_num))
        cols = int(np.ceil(cluster_num / rows))

        # Create subplot mosaic with appropriate dimensions
        mosaic_layout = [list(range(i * cols, min((i + 1) * cols, cluster_num))) for i in range(rows)]
        fig, axes_dict = plt.subplot_mosaic(mosaic_layout, figsize=(cols * 5, rows * 5))

        for i, specs in enumerate(specs_list):
            ax = axes_dict[i]
            gated_x, gated_y, colors, labels = self.process_data(specs)

            if specs.graph_type == 'scatter':
                all_x = np.concatenate(gated_x)
                all_y = np.concatenate(gated_y)
                all_colors = np.concatenate([[colors[j]] * len(gated_x[j]) for j in range(len(gated_x))])
                ax.scatter(all_x, all_y, c=all_colors, s=specs.point_size)
                ax.set_xlabel(specs.x_label)
                ax.set_ylabel(specs.y_label)
                ax.set_title(specs.title)
                ax.legend(labels)
            elif specs.graph_type == '2D_histogram':
                ax.hist2d(np.concatenate(gated_x), np.concatenate(gated_y), bins=specs.bins, cmap='viridis')
                ax.set_xlabel(specs.x_label)
                ax.set_ylabel(specs.y_label)
                ax.set_title(specs.title)

        # Remove any unused subplots
        for j in range(len(axes_dict), rows * cols):
            fig.delaxes(axes_dict[j])

        plt.tight_layout()
        plt.show()

