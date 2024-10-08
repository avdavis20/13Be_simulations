# ELOSS CALCULATION PROGRAM
# created by: Alyssa Davis, Summer 2024

# PROGRAM PURPOSE: Reads in an input h5 file full of simulated kinematic information
#   (acquired by running reaction through attpc-engine pipeline), then analyzes 
#   whether or not simulated reaction events will be measurable by detectors. Outputs
#   an h5 file containing the following simulation results: angular acceptances, event 
#   visibility flags, 

# USAGE: "python3 eloss_calc.py insert_file_name.h5"

# OPTIONS:
#  -a, --avail               (Optional) Print list of all event numbers available
#                            in provided input file.
#  -e, --eventselect TEXT    (Optional) Choose specific event numbers for analysis,
#                            inputted as a string of numbers separated by commas.
#  -n, --nucleusselect TEXT  (Optional) Choose specific nucleonic constituents to
#                            examine.
#  -v, --verbose             (Optional) Toggle individual events/nuclei
#                            print statements.
#  -w, --write               (Optional) Write data to h5 file.

#############################################################################################################
#############################################################################################################

# IMPORT LIBRARIES
from writer.data_writer import Event, dataWriter, Analyzer
from attpc_engine import nuclear_map 
from pathlib import Path
import spyral_utils.nuclear.target as spytar 
import spyral_utils.nuclear as spynuc 
import numpy as np
import sys
import json
import vector
import click
import h5py 

#############################################################################################################
# INITIALIZATION OF CONSTANTS/CONVENTIONS/CONFIG VARIABLES
#############################################################################################################

c_const = 1   # calculations done in natural units 

config_path = Path("/Users/Owner/Desktop/AT-TPC Research/13Be_simulations/config.json")
with open(config_path, 'r') as f: config = json.load(f)     # load config file

in_dir = config['workspace']['kin_output_path']             # input dir
out_dir = config['workspace']['eloss_output_path']          # output dir

nucl_dict = {'t': 'TARGET', 'p': 'PROJECTILE', 'e': 'EJECTILE', 'r': 'RESIDUAL'}  # nucl convention

Z = [list(config['rxn_info'].items())[i][1] for i in range(1,8,2)]     # proton numbers
A = [list(config['rxn_info'].items())[i][1] for i in range(2,9,2)]     # mass numbers

latex = [nuclear_map.get_data(Z[i],A[i]).isotopic_symbol for i in range(0,4)]  # nucl latex

tar_data = spytar.TargetData(
    compound = config['target']['compound'],
    pressure = config['target']['pressure'],
    thickness = config['target']['thickness']
)
targ = spytar.GasTarget(tar_data, spynuc.NuclearDataMap())   # targ definition
proj = nuclear_map.get_data(Z[0],  A[0])                     # proj definition

R = config['detector_geometry']['R']     # radius of detector
H = config['detector_geometry']['H']     # height of detector     

#############################################################################################################

def detection_test(X0,P,range_):
    """
    Parameterizes the trajectory of particle and applies geometric detector constraints. 
    Determines whether or not particle will be detected.

    Params:
    -- X0 [float]: Event vertex in R3.
    -- P [vector.MomentumNumpy4D]: Array of 4-momentum vectors for all 4 particles in event.
    -- range_ [float]: Array of particles' stopping distances within detector medium.
    
    Returns: 
    -- is_detected (Bool): flag indicating if particle was detected.
    """

    # Parametrized particle trajectory:
    # x(t) = x0 + vx * t
    # y(t) = y0 + vy * t
    # z(t) = z0 + vz * t

    # Cylindrical surface boundary constraint:
    # x(t)^2 + y(t)^2 = R^2
    # 0 < z(t) < H


    # Split 4-momentum vector into 3-momentum and energy
    p_ = np.array([np.array(P["px"]), np.array(P["py"]), np.array(P["pz"])]) # array of particles' 3-momenta, in MeV/c
    E = np.array(P["E"]) # array of particles' energies, in MeV
    [x0,y0,z0] = X0 # particles' starting location

    # Calculate the particles' velocities in the lab frame
    V_lab = [vx,vy,vz] = (p_ * c_const**2) / (E) 

    # Calculate roots of parameterized equations of motion with applied constraints
    a = vx**2 + vy**2
    b = 2 * (vx + vy)
    c = x0**2 + y0**2 - R**2

    # Calculate discriminant
    disc = b**2 - (4*a*c)
    
    # Find roots of polynomial
    root1 = []
    root2 = []
    for i in range(len(a)):
        # if trajectory doesn't intersect with cylinder wall... (aka, no x-y motion)
        if a[i]==0:
            root1.append((-z0)/vz[i])
            root2.append((H-z0)/vz[i])
        # ... calculate inntersection time at the caps.
        else: 
            root1.append(-b[i] + (np.sqrt(disc[i]) / (2.0 * a[i])))
            root2.append(-b[i] - (np.sqrt(disc[i]) / (2.0 * a[i])))

    # Remove the root that is negative, as its parametric trajectory is in the opposite direction
    t = []
    for i in range(len(root1)):
        t_values = [t for t in [root1[i], root2[i]] if t > 0]
        t.append(min(t_values))
        if not t_values:
            t.append(000.)
            raise ValueError("Geometrical Constraint Error: No positive t-values found, indicating no valid intersection in the trajectory direction.")

    # Calculate intersection point
    x_intersect = x0 + vx * t
    y_intersect = y0 + vy * t
    z_intersect = z0 + vz * t    

    # Calculate minimized geometric distance btwn X0 and the constraint boundaries
    geo_dist = np.sqrt((x_intersect - x0)**2 + (y_intersect - y0)**2 + (z_intersect - z0)**2)

    # Determine if particle will be detected based on geometric allowance of range distance
    is_detected = []
    for i in range(len(range_)):
        is_detected.append (True) if geo_dist[i] > range_[i] else is_detected.append(False)

    print('\n\n\nDETECTION TEST RESULTS...\n\n')
    print(f'Reaction Vertex:\n{X0}\n')
    print(f'4-momentum ([px,py,pz,E]_i):\n{P}\n')
    print(f'3-momentum ([px_i,py_i,pz_i]):\n{p_}\n')
    print(f'Energy ([E_i]):\n{E}\n')
    print(f'Lab velocity ([vx_i,vy_i,vz_i]):\n{V_lab}\n')
    print(f'a, b, c of polynomial roots calculation:\n{a}\n{b}\n{c}\n')
    print(f'Parameterized intersection time btwn particle trajectory and constraints: \n{t}\n')
    print(f'Particle detected?\n{is_detected}\n\n\n')
    
    return is_detected

#############################################################################################################

def calc_angles(P_lab):
    """ Calculates polar and azimuthal angles for each particle of a reaction event,
    from both lab and center of mass frames.
    
    Params: 
    - P [vector.MomentumNumpy4D]: Array of 4-momentum vectors for all 4 particles in event.

    Returns: 
    - angles_lab [[float],[float]]: polar and azimuthal angles in lab frame (in deg)
    - angles_cm [[float],[float]]: polar and azimuthal angles in cm frame (in deg)
    """

    # Calculate initial total 4-momentum in the lab frame
    Ptot_lab: vector.MomentumObject4D = (P_lab[0]+P_lab[1])

    # Boost total 4-momentum array into center of mass frame
    P_cm: vector.MomentumObject4D = P_lab.boostCM_of(Ptot_lab)

    # Calculate polar/azimuthal angles of particle trajectories in both frames
    polar_lab = np.rad2deg(np.array(P_lab.rho))
    azi_lab = np.rad2deg(np.array(P_lab.phi))
    angles_lab = [polar_lab,azi_lab]

    polar_cm = np.rad2deg(np.array(P_cm.rho))
    azi_cm = np.rad2deg(np.array(P_cm.phi))
    angles_cm = [polar_cm,azi_cm]
    
    print('\n\n\nANGLE CALCULATION RESULTS...\n\n')
    print(f'Lab Frame 4-vectors: \n{P_lab}\n')
    print(f'CM Frame 4-vectors: \n{P_cm}\n')
    print(f'LAB ANGLES:\n   Polar: {angles_lab[0]}\n   Azimuthal: {angles_lab[1]}\n')
    print(f'CM ANGLES:\n   Polar: {angles_cm[0]}\n   Azimuthal: {angles_cm[1]}')

    return angles_lab, angles_cm

#############################################################################################################

@click.command()
@click.argument('sim_file')
@click.option('-a', '--avail', is_flag=True, default=False, help='(Optional) Print list of all event numbers available in provided input file.')
@click.option('-e', '--eventselect', default='all', help='(Optional) Choose specific event numbers for analysis, inputted as a string of numbers separated by commas.')
@click.option('-n', '--nuclselect', default='t,p,e,r', help='(Optional) Choose specific nucleonic constituents to examine.')
@click.option('-v', '--verbose', is_flag=True, default=False, help='(Optional) Toggle individual events/nuclei print statements.')
@click.option('-w','--write', is_flag=True, default=False, help='(Optional) Write data to h5 file.')


def main(sim_file, avail, eventselect, nuclselect, verbose, write):

    # DEFINE INPUT/OUTPUT PATHS
    eloss_file = f'eloss{sim_file[3:]}'
    in_path = Path(f'{in_dir}/{sim_file}')
    out_path = Path(f'{out_dir}/{eloss_file}')

    with h5py.File(in_path,'r') as hdf:     
    
        # Initialize analyzer class and print program greeting
        init_info = {
            'latex': latex,
            'nucl_dict': nucl_dict,
            'in_path': in_path,
            'out_path': out_path
        }
        analyzer = Analyzer(init_info, verbose)
        analyzer.welcome()

        # Configure analysis setup according to any optional flags raised in command line
        config_package = {
            'avail': avail,
            'eventselect': eventselect,
            'nuclselect': nuclselect, 
            'data': hdf['/data'],
            'write': write,
            'filename': eloss_file,
            'run_num': sim_file[4:8]
        }
        selected_events, selected_nuclei = analyzer.config_options(config_package)

        
        # Print reaction information
        rxn_info = {
            "events": selected_events,
            "nuclei": selected_nuclei,
            "kindir": config['rxn_info']['inverse_kinematics']
        }
        analyzer.print_rxn_info(rxn_info)


        # CYCLE THROUGH ANALYSIS FOR EACH EVENT
        for event_name in selected_events:
            # Get the event data from the h5 file
            event_data = hdf[f'/data/event_{event_name}'] 
            event_info = {
                'event_data': event_data,
                'event_name': event_name,
            }
            X0,P_lab = analyzer.get_event_info(event_info)
            calc_angles(P_lab)



            

            #print(f"/////////////////////////////////   EVENT #{event_name}   ///////////////////////////////////// \n",verbose)
            


            




    ########
    #last stopped working here
    ########

    # Define reaction vertex
    #X0 = np.array([0.00663388, -0.00138931, 0.44978139]) 

    # Define particle ranges
    #range_ = np.array([0.0, 0.0, 0.0, 2.5235470545983882])
    
    # Define vector object array for 4-momentum test vectors in lab frame
    #P_lab: vector.MomentumObject4D = vector.array({
    #    "px": [0., 0., 0., 1875.61292879],
    #    "py": [0., 0., 1233.50835481, 11268.67848734],
    #    "pz": [51.25162181, 93.21752461, 59.42052789, 946.15089738],
    #    "E": [51.25162181, 93.21752461, 1174.08782692, 12198.14051874]
    #})


if __name__ == '__main__':
    main()