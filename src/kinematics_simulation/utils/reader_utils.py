import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Dict

class dataReader:
    def __init__(self, file_path: str, data_profile: str):
        self.file_path = Path(file_path) # Data file location
        self.data_profile = data_profile # What sort of data we are looking at

        self.eloss_data: Optional[Dict] = None
        self.trace_data: Optional[Dict] = None

    def read_eloss_data(self):
        """
        Reads in the data from the eloss results hdf5 file to be plotted.
        """
        with h5py.File(self.file_path, 'r') as f:
            # Access the run group (assuming a single run group)
            run_group_name = list(f.keys())[0]
            run_group = f[run_group_name]

            # Initialize arrays to collect data across all events
            event_KEs = []
            polar_lab_angles = []
            polar_cm_angles = []
            azimuthal_lab_angles = []
            azimuthal_cm_angles = []
            vx_values = []
            vy_values = []
            vz_values = []
            z_positions = []
            is_detected = []
            isotopes = []

            # Loop over events and collect data
            for event_key in run_group.keys():
                event = run_group[event_key]

                # Collect Kinetic Energies
                if 'event_KEs' in event:
                    event_KEs.extend(event['event_KEs'][:])
                else:
                    print(f"Warning: 'event_KEs' not found in {event_key}")

                # Collect Lab Angles
                if 'angles_lab' in event and 'polar' in event['angles_lab'] and 'azimuthal' in event['angles_lab']:
                    polar_lab_angles.extend(event['angles_lab/polar'][:])
                    azimuthal_lab_angles.extend(event['angles_lab/azimuthal'][:])
                else:
                    print(f"Warning: 'angles_lab' or 'polar'/'azimuthal' not found in {event_key}")
                
                # Collect CM Angles
                if 'angles_cm' in event and 'polar' in event['angles_cm'] and 'azimuthal' in event['angles_cm']:
                    polar_cm_angles.extend(event['angles_cm/polar'][:])
                    azimuthal_cm_angles.extend(event['angles_cm/azimuthal'][:])
                else:
                    print(f"Warning: 'angles_cm' or 'polar'/'azimuthal' not found in {event_key}")

                # Collect Velocities
                if 'velocities' in event:
                    vx_values.extend(event['velocities/vx'][:])
                    vy_values.extend(event['velocities/vy'][:])
                    vz_values.extend(event['velocities/vz'][:])
                else: 
                    print(f"Warning: 'velocities' not found in {event_key}")

                # Collect Detection Status and convert to bool type
                is_detected.extend(event['is_detected'][:])

                # Collect Event Vertex (Z position)
                if 'event_vertex' in event.attrs:
                    z_positions.append(event.attrs['event_vertex'][2])

                # Collect Isotopes (Target, Projectile, Ejectile, Residual)
                isotopes.append(event['isotopes'][:])
            
            # Convert lists to numpy arrays for easier handling
            eloss_data = {
                'event_KEs': np.array(event_KEs),
                'polar_lab_angles': np.array(polar_lab_angles),
                'polar_cm_angles': np.array(polar_cm_angles),
                'azimuthal_lab_angles': np.array(azimuthal_lab_angles),
                'azimuthal_cm_angles': np.array(azimuthal_cm_angles),
                'vx_values': np.array(vx_values),
                'vy_values': np.array(vy_values),
                'vz_values': np.array(vz_values),
                'z_positions': np.array(z_positions),
                'is_detected': np.array(is_detected, dtype=bool),
                'isotopes': np.array(isotopes, dtype=str)
            }
            
            self.eloss_data = eloss_data
            return self.eloss_data

    def read_trace_data(self):
        """
        Reads in the data from the Si trace data HDF5 file to be plotted.
        """
        with h5py.File(self.file_path, 'r') as hdf:
            events_group = hdf['events']
            trace_data = {}

            for event_key in events_group:
                if 'event_' in event_key:
                    event_group = events_group[event_key]
                    if 'get_traces' in event_group:
                        get_traces_dset = event_group['get_traces']
                        trace_data[event_key] = get_traces_dset[()]
            
            self.trace_data = trace_data
            return self.trace_data

################################# PRINT STATEMENT METHODS #################################

    def print_all_trace_info(self):
        # Remove 'event_0' if it exists, assuming it might just be metadata
        if 'event_0' in self.trace_data:
            del self.trace_data['event_0']
            print("'event_0' ignored as it contains no real data.")

        # Number of events in the dataset
        num_events = len(self.trace_data)
        print(f"Number of events loaded (excluding event_0): {num_events}")

        # Checking the structure of one of the events to verify its shape
        sample_event_key = list(self.trace_data.keys())[0] if num_events > 0 else None
        if sample_event_key:
            sample_event_data = self.trace_data[sample_event_key]
            print(f"Structure of sample event '{sample_event_key}':")
            print(f" - Data Shape: {sample_event_data.shape}")
            print(f" - Data Type: {sample_event_data.dtype}")

        # Calculate how many events have data and how many are empty
        events_with_data = [key for key, data in self.trace_data.items() if data.size > 0]
        events_without_data = [key for key, data in self.trace_data.items() if data.size == 0]

        print(f"Number of events with valid data: {len(events_with_data)}")
        print(f"Number of events without valid data: {len(events_without_data)}")

    def print_all_eloss_info(self):
        """
        Function to cleanly print all information from the eloss HDF5 file.
        """
        # Open the HDF5 file
        with h5py.File(self.file_path, 'r') as f:
            # Access the run group
            run_group = f[list(f.keys())[0]]
            
            # Access the event names
            event_names = [i[6:] for i in list(run_group.keys())]
            
            # Calculate number of event entries
            num_event_entries = len(event_names)
            num_particle_entries = num_event_entries * 4
        
            # Print out all the events and their information
            print("\n" + "-"*80)
            print(f"{'Event':<10}{'Isotope':<15}{'Polar Lab Angle':<20}{'Kinetic Energy':<20}{'Detected':<10}")
            print("-"*80)
            for i in range(num_event_entries):
                print(f"Event {event_names[i]}")
                for j in range(4):
                    print(f"{' ':<10}{self.read_eloss_data()['isotopes'][i][j]:<15}{self.read_eloss_data()['polar_lab_angles'][i]:<20.3f}{self.read_eloss_data()['event_KEs'][j]:<20.3f}{self.read_eloss_data()['is_detected'][j]:<10}")
            print("-" * 80)
    
    def print_event_info(self, event_number=100):
        """
        Print all the data for a specific event to ensure proper correlation between variables.
        Params:
        - event_number: The event number to print. Default is 100.
        """

        if self.data_profile == 'eloss':
            # Define the number of particles per event (target, projectile, ejectile, residual)
            chunk_size = 4

            # Check if the event number is valid
            if event_number < 1 or event_number > len(self.read_eloss_data()['isotopes']):
                print(f"Invalid event number. Please choose another event number.")
                return

            # Adjust the index to be zero-based
            event_idx = event_number - 1
            
            print(f"{'Event':<8} {'Isotope':<10} {'Polar Lab Angle':<20} {'Kinetic Energy':<20} {'Detected':<10}")
            print("-" * 70)
            print(f"Event {event_number}")
            
            for j in range(chunk_size):
                idx = event_idx * chunk_size + j
                # Decode the byte string
                isotope_str = self.read_eloss_data()['isotopes'][event_idx][j]
                print(f"{'':<8} {isotope_str:<10} {self.read_eloss_data()['polar_lab_angles'][idx]:<20.3f} {self.read_eloss_data()['event_KEs'][idx]:<20.3f} {self.read_eloss_data()['is_detected'][idx]:<10}")
            
            print("-" * 70)
