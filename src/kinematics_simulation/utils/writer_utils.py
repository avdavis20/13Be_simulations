from dataclasses import dataclass 
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path


@dataclass
class EventData:
    event_num: int  # Event number
    is_detected: list  # Particle detection flags (list of booleans)
    event_KEs: list  # List of kinetic energies (list of floats)
    event_vertex: list  # Event vertex (list of floats, representing x, y, z coordinates)
    isotopes: list  # List of isotopes involved in the rxn event (list of strings)
    angles_lab: dict  # Dictionary of lab angles {'polar': [float], 'azimuthal': [float]}
    angles_cm: dict  # Dictionary of CM angles {'polar': [float], 'azimuthal': [float]}
    velocities: dict  # Dictionary with keys 'vx', 'vy', 'vz' and their corresponding lists (arrays for each velocity component)

class dataWriter:
    def __init__(self, out_dir, filename, run_num, verbose, debugWrite):
        self.out_dir = out_dir
        self.filename = filename
        self.run_num = run_num
        self.verbose = verbose
        self.debugWrite = debugWrite
    
    def save_plots_to_pdf(self, grapher, filename='trace_signals_summary.pdf'):
        """
        Save plots of all trace events to a PDF, arranged in a compact grid layout.
        """
        try:
            # Ensure output directory is a Path object
            out_dir_path = Path(self.out_dir)

            # Define the output path for the PDF
            pdf_path = out_dir_path / filename

            # Delete the old PDF if it exists
            if pdf_path.exists():
                pdf_path.unlink()

            # Create a new PDF file and plot all events in a grid format
            with PdfPages(pdf_path) as pdf:
                skipped_events = []
                print(f"Plotting signals for all {len(grapher.data)} events...")

                # Initialize the figure for grid layout
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 3 rows x 4 columns for 12 graphs per page
                axes = axes.flatten()
                current_plot = 0

                for event_key, event_data in tqdm(grapher.data.items()):
                    if event_data.shape[0] < 2:
                        skipped_events.append(event_key)
                        continue

                    # Plot all available directions for the event on the same graph
                    ax = axes[current_plot]
                    for i in range(event_data.shape[0]):
                        ax.plot(event_data[i, :], label=f"Direction {i + 1}")

                    ax.set_xlabel("Time Bins")
                    ax.set_ylabel("Signal Amplitude")
                    ax.set_title(f"Event: {event_key}")
                    ax.legend()
                    ax.grid(True)

                    current_plot += 1

                    # If the grid is full, save the page and reset for the next set of plots
                    if current_plot == len(axes):
                        plt.tight_layout()
                        pdf.savefig(fig)  # Save the current figure (page) into the PDF
                        plt.close(fig)

                        # Create a new figure for the next set of plots
                        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
                        axes = axes.flatten()
                        current_plot = 0

                # Save the remaining plots if there are any
                if current_plot > 0:
                    for i in range(current_plot, len(axes)):
                        fig.delaxes(axes[i])  # Remove unused subplots
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                print("Finished plotting all events.")
                if skipped_events:
                    print(f"{len(skipped_events)} skipped events due to insufficient data shape: {skipped_events}")

        except Exception as e:
            print(f"An error occurred while saving plots to PDF: {e}")
        
    def write_data(self, event: EventData):
        """
        Writes a single event to the HDF5 file, appending the event to the file.
        """
        try:
            # Open the file in append mode ('a') - this makes it so that previous event's data isnt written over
            with h5py.File(self.out_dir/self.filename, 'a') as f:
                # Check if the run group already exists; if not, create it
                if f.get(f'run_{self.run_num}') is None:
                    run_group = f.create_group(f'run_{self.run_num}')
                else:
                    run_group = f[f'run_{self.run_num}']
                
                # Check if the event group already exists, and remove it if needed
                event_group_name = f'event_{event.event_num}'
                if event_group_name in run_group:
                    del run_group[event_group_name]  # Optionally delete existing group
                
                # Create a group for the new event within the run
                event_group = run_group.create_group(f'event_{event.event_num}')

                # Store event vertex as event attribute 
                if self.debugWrite:
                    print(f"Storing event vertex for event {event.event_num}: {event.event_vertex}")
                event_group.attrs['event_vertex'] = np.array(event.event_vertex)

                # Convert isotopes list to numpy array with variable-length string type
                vlen_str_dtype = h5py.special_dtype(vlen=str)
                isotopes_array = np.array(event.isotopes, dtype=vlen_str_dtype)  # Convert to numpy array with proper dtype
                if self.debugWrite:
                    print(f"Storing isotopes for event {event.event_num}: {event.isotopes}")
                event_group.create_dataset('isotopes', data=isotopes_array)

                # Store detection results (boolean list)
                if self.debugWrite:
                    print(f"Storing detection results for event {event.event_num}: {event.is_detected}")
                event_group.create_dataset('is_detected', data=np.array(event.is_detected, dtype=bool))

                # Store kinetic energies (list of floats)
                if self.debugWrite:
                    print(f"Storing kinetic energies for event {event.event_num}: {event.event_KEs}")
                event_group.create_dataset('event_KEs', data=np.array(event.event_KEs))

                # Store velocities in a group (using vector.array) if they are not empty
                velocities_group = event_group.create_group('velocities')
                if len(event.velocities['vx']) > 0:
                    if self.debugWrite:
                        print(f"Storing velocities['vx'] for event {event.event_num}")
                    velocities_group.create_dataset('vx', data=np.array(event.velocities['vx']))
                else:
                    if self.debugWrite:
                        print(f"Warning: velocities['vx'] is empty for event {event.event_num}")

                if len(event.velocities['vy']) > 0:
                    velocities_group.create_dataset('vy', data=np.array(event.velocities['vy']))
                if len(event.velocities['vz']) > 0:
                    velocities_group.create_dataset('vz', data=np.array(event.velocities['vz']))

                # Store angles in lab frame (polar and azimuthal) if they are not empty
                angles_lab_group = event_group.create_group('angles_lab')
                if len(event.angles_lab['polar']) > 0:
                    if self.debugWrite:
                        print(f"Storing angles_lab['polar'] for event {event.event_num}")
                    angles_lab_group.create_dataset('polar', data=np.array(event.angles_lab['polar']))
                if len(event.angles_lab['azimuthal']) > 0:
                    angles_lab_group.create_dataset('azimuthal', data=np.array(event.angles_lab['azimuthal']))

                # Store angles in center of mass frame (polar and azimuthal) if they are not empty
                angles_cm_group = event_group.create_group('angles_cm')
                if len(event.angles_cm['polar']) > 0:
                    angles_cm_group.create_dataset('polar', data=np.array(event.angles_cm['polar']))
                if len(event.angles_cm['azimuthal']) > 0:
                    angles_cm_group.create_dataset('azimuthal', data=np.array(event.angles_cm['azimuthal']))

            if self.debugWrite:
                print(f"Event {event.event_num} successfully written to '{self.filename}'.")

        except Exception as e:
            print(f"An error occurred while writing data: {e}")
 

