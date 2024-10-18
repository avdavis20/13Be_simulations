""" ELOSS CALCULATION PROGRAM
created by: Alyssa Davis, 10/18/2024

PROGRAM PURPOSE: 
    Reads an input h5 file full of simulated kinematic information
    (acquired by running reaction through attpc-engine pipeline), then 
    conducts various energy loss calculations and determines whether
    events will be detected by AT-TPC.

OUTPUT H5 FILE STRUCTURE:
    eloss_<run_num>.h5
    └── run_<run_num>/
        ├── event_<event_num>/
        │   ├── event_vertex (attribute) [x0, y0, z0]
        │   ├── isotopes (dataset) [list of isotope strings]
        │   ├── is_detected (dataset) [boolean array]
        │   ├── event_KEs (dataset) [float array]
        │   ├── velocities/
        │   │   ├── vx (dataset) [float array]
        │   │   ├── vy (dataset) [float array]
        │   │   └── vz (dataset) [float array]
        │   ├── angles_lab/
        │   │   ├── polar (dataset) [float array]
        │   │   └── azimuthal (dataset) [float array]
        │   └── angles_cm/
        │       ├── polar (dataset) [float array]
        │       └── azimuthal (dataset) [float array]

COMMAND LINE USAGE: 
    "python3 trajectory_analysis.py <insert_sim_file_name.h5> <optional flags>"

FLAG OPTIONS:
  -a,   --avail             (Optional) Print list of all event numbers available
                            in provided simulation file.
  -e,   --eventselect       (Optional) Choose specific event numbers for analysis,
                            input as a string of numbers separated by commas.
  -n,   --nucleusselect     (Optional) Choose specific nuclei to examine, input
                            as t,p,e,r options separated by commas in a string.
  -v,   --verbose           (Optional) Toggle individual event print statements.
  -w,   --write             (Optional) Write data to h5 file.
  -dd,  --debugdetect       (Optional) Print out all detection test calculations.
  -da,  --debugangles       (Optional) Print out all angle calculations.
  -dw,  --debugwrite        (Optional) Print out all write updates.
"""

#############################################################################################################
#############################################################################################################

from utils.eloss_utils import elossUtils
from utils.writer_utils import EventData
from pathlib import Path
from tqdm import tqdm
import click
import h5py 

@click.command()
@click.argument('sim_file')
@click.option('-a',  '--avail', is_flag=True, default=False, help='(Optional) Print list of all event numbers available in provided input file.')
@click.option('-e',  '--eventselect', default='all', help='(Optional) Choose specific event numbers for analysis, entered as a string of numbers separated by commas.')
@click.option('-n',  '--nuclselect', default='t,p,e,r', help='(Optional) Choose specific isotopes to analyze, entered as t,p,e,r options separated by commas.')
@click.option('-v',  '--verbose', is_flag=True, default=False, help='(Optional) Toggle individual event rxn info statements.')
@click.option('-w',  '--write', is_flag=True, default=False, help='(Optional) Write data to h5 file.')
@click.option('-dd', '--debugdetect', is_flag=True, default=False, help='(Optional) Toggle detection algorithm print statements.')
@click.option('-da', '--debugangles', is_flag=True, default=False, help='(Optional) Toggle angle calculator print statements.')
@click.option('-dw', '--debugwrite', is_flag=True, default=False, help='(Optional) Toggle writer class print statements.')

def main(sim_file, avail, eventselect, nuclselect, verbose, write, debugdetect, debugangles, debugwrite):

    # Initialize analysis utils 
    analyzer = elossUtils()

    # Configure data given in config file
    config_path = Path("/Users/Owner/Desktop/AT-TPC Research/13Be_simulations/config.json")
    analyzer.get_config_data(config_path)
    
    # Open kinematics file
    with h5py.File(f'{analyzer.in_dir}/{sim_file}','r') as hdf:    

        # Send program welcome message
        analyzer.welcome()

        # Configure flag settings
        flags = {
            'avail': avail,
            'eventselect': eventselect,
            'nuclselect': nuclselect,
            'verbose': verbose,
            'write': write,
            'debugDetect': debugdetect,
            'debugAngles': debugangles,
            'debugWrite': debugwrite,
            'in_filename': sim_file,
            'out_filename': Path(f'eloss{sim_file[3:]}'),
            'run_num': sim_file[4:8], 
            'eventnum_data': hdf['/data']
        }
        analyzer.config_options(flags)

        # Locally store analysis scope defined by user flag input
        selected_events = analyzer.selected_events
        
        # Print reaction information
        analyzer.print_rxn_info()
        
        # Initialize tqdm progress bar
        total_steps = len(selected_events) * 2  # Two steps for each event: (1) calculation, (2) writing
        
        with tqdm(total=total_steps, desc="Processing and Writing Events", unit="step") as pbar:

            # Cycle through analysis/writing process for each event
            for event_name in selected_events:

                # Get the event data from the h5 file
                event_data = hdf[f'/data/event_{event_name}'] 
                event_info = {
                    'event_data': event_data,
                    'event_name': event_name,
                }

                analyzer.get_event_info(event_info)

                # STEP 1: CALCULATIONS ~
                # Calculate each particle's kinetic energy 
                analyzer.calc_KEs(analyzer.P_lab)
            
                # Run the detection test
                is_detected = analyzer.test_detection(analyzer.X0, analyzer.P_lab)

                # Calculate each partile's angular distribution
                analyzer.calc_angles(analyzer.P_lab)

                # Store results in a dictionary parallel to EventData dataclass format
                event_results = {
                    'event_num': event_name,
                    'is_detected': is_detected,
                    'event_KEs': analyzer.event_KEs,
                    'event_vertex': analyzer.X0,
                    'isotopes': analyzer.latex,
                    'angles_lab': {
                        'polar': analyzer.angles_lab[0],
                        'azimuthal': analyzer.angles_lab[1]
                    },
                    'angles_cm': {
                        'polar': analyzer.angles_cm[0],
                        'azimuthal': analyzer.angles_cm[1]
                    } ,
                    'velocities': {
                        'vx': analyzer.V_lab['x'],
                        'vy': analyzer.V_lab['y'],
                        'vz': analyzer.V_lab['z']
                    }
                }
                # Step 1 complete, update the progress bar
                if write: 
                    pbar.update(1)
                else: 
                    pbar.update(2)

                # STEP 2: WRITE RESULTS TO FILE ~
                if write: 
                    event_data = EventData(**event_results)
                    analyzer.writer.write_data(event_data)
                    
                    # Step 2 complete, update the progress bar
                    pbar.update(1)  

            
if __name__ == '__main__':
    main()