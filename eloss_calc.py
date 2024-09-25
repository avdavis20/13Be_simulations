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
import numpy as np
import pandas as pd
import json
import sys
import h5py # type: ignore
import click # type: ignore
import spyral_utils.nuclear.target as spytar # type: ignore
import spyral_utils.nuclear as spynuc # type: ignore
import matplotlib.pyplot as plt #type: ignore
from attpc_engine import nuclear_map # type: ignore
from pathlib import Path
from decimal import Decimal
from data_writer import Event, dataWriter

# SOME MISC. CONSTANT INITIALIZATIONS... 
c_const = 299792458.0 #speed of light in m/s
avo_const = 1.66054E-27 #amu to kg conversion
rad2deg = 360/(2*np.pi) #rad to deg conversion

#############################################################################################################
# FUNCTION DEFINITIONS
#############################################################################################################

# ADMIN FUNCTIONS..................

# Welcome Message
def welcome():
    print('\nWELCOME TO THE ENERGY LOSS CALCULATOR.\n')

# Toggle print
def cprint(s,v):
    if v:
        print(s)
    return

# Truncate scientific notation decimals for print statements
def format_e(n):
    a = '%E' % Decimal(n)
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

# Read in event numbers from hdf5 file
def get_eventnums(hdf):
    """
    Creates a list of all available event numbers in hdf5 file.
    """
    event_nums = []
    for key in hdf['/data']:
        event_nums.append(key.split('_')[-1])
    print(event_nums)
    sys.exit()

#############################################################################################################

# SIMPLE CALCULATIONS................

# Extract kinetic energy from relativistic energy spread value
def getKE(Z,A,rel_E):
    """
    Calculates kinetic energy in MeV from given relativistic energy spread. 
    Uses tabulated atomic mass values. 

    Parameters: 

    - Z: proton number
    - A: mass number 
    - rel_E: relativistic energy value (from hdf5 data)

    Returns:

    - KE: kinetic energy value
    """
    nuclear_map.get_data(Z,A)
    KE = rel_E - (nuclear_map.get_data(Z,A).atomic_mass * 931.494)

    return KE # in MeV

# Calculate velocity V from momentum P, relativistic energy E
def getV(P,E):
    """
    Calculates velocity components from momenta and total relativistic energy value.

    Parameters:

    - P (np array of floats): [Px,Py,Pz]
    - E (float): relativistic energy

    Returns:

    - V (np array of floats): [vx,vy,vz]
    """
    #m = nuclear_map.get_data(Z, A).atomic_mass * avo_const # rest mass in kg
    P_conv = P * 1.60218E-13 / c_const # convert P from MeV to kg m/s
    P_mag = np.sqrt(np.sum(P_conv**2)) # momentum magnitude
    V_mag = (P_mag * c_const**2)/E # velocity magnitude
    vx = V_mag * (P_conv[0]/P_mag)
    vy = V_mag * (P_conv[1]/P_mag)
    vz = V_mag * (P_conv[2]/P_mag)
    
    return np.array([vx,vy,vz])

#############################################################################################################

# TEST TO DETERMINE WHETHER OR NOT WE CAN SEE AN EVENT IN DETECTOR................

def test_event_validity(X,V,range_,R,H,v):
    """
    Calculates the angle of reaction trajectory and tests to see if we will 
    be able to measure it from a detector depending on incident energies. 

    Parameters:

    - X = [x0, y0, z0] (list of floats): reaction vertex in R3
    - V = [vx, vy, vz] (list of floats): initial particle velocity in R3
    - range_ (float): range of particle

    Returns:

    - angle (float): angle of trajectory, 
    - intersect_data (tuple): x,y,z intersections, and time of intersection if
        applicable
    - is_valid (bool): whether or not we will detect the simulated 
        reaction event
    """
    x0, y0, z0 = X[0], X[1], X[2]
    vx, vy, vz = V[0], V[1], V[2]

    # Parametric equations of the particle's path:
    # x(t) = x0 + vx * t
    # y(t) = y0 + vy * t
    # z(t) = z0 + vz * t

    # Cylindrical surface boundary:
    # x^2 + y^2 = R^2

    # Finding the quadratic roots:
    a = vx**2.0 + vy**2.0
    b = 2.0 * ((x0 * vx) + (y0 * vy))
    c = x0**2.0 + y0**2.0 - (R)**2.0

    cprint(f'\nCalculating discriminant using\n      a = {a:3f},\n      b = {b:3f},\n      c = {c:3f}...',v)
    discriminant = (b**2.0) - (4.0*a*c)
    cprint(f'\nDiscriminant Result:\n      {format_e(discriminant)}',v)

    if discriminant < 0.: 
        raise ValueError("No real solutions for intersection, discriminant < 0")
    
    # Calculate the times at which the trajectory hits geometric boundary, if at all
    if a == 0.0:
        cprint("\nParticle does not have x-y motion.\n",v)
        xymotion = False
        t1,t2 = ((-z0)/vz) , ((H-z0)/vz)
    else:
        cprint("\nParticle has x-y motion.\n",v)
        xymotion = True
        t1,t2 = (-b + (np.sqrt(discriminant) / (2.0*a))), (-b - (np.sqrt(discriminant) / (2.0*a)))
        

    cprint(f"t values:\n t1 = {t1},\n t2 = {t2}",v)

    # Remove the t that is negative, as its parametric trajectory was in the wrong direction
    t_values = [t for t in [t1, t2] if t > 0]
    if not t_values:
        raise ValueError("No positive t values found, indicating no valid intersection in the trajectory direction.")
    t = min(t_values)
    
    # Calculate intersection point
    x_intersect = x0 + vx * t
    y_intersect = y0 + vy * t
    z_intersect = z0 + vz * t

    # Pack intersection data into tuple
    intersect_data = [x_intersect,y_intersect,z_intersect]
    
    # Calculate geometric distance between starting point and closest point on geometric boundary
    geo_dist = np.sqrt((x_intersect - x0)**2 + (y_intersect - y0)**2 + (z_intersect - z0)**2)

    # calculate post-reaction trajectory angle of nucleus of interest
    if geo_dist != 0:
        angle_lab = np.arccos(vz / geo_dist) * rad2deg
        angle_cm = 999
    else:
        angle_lab = 999
        angle_cm = 999
        print(f"Warning: Zero geometric distance encountered for event at ({x0}, {y0}, {z0}) with velocities ({vx}, {vy}, {vz}).")
        is_valid = False

    # Check if it'll hit the wall 
    if geo_dist > range_:
        is_valid = True
    else: 
        is_valid = False

    return angle_lab, angle_cm, is_valid

#############################################################################################################
# USAGE
#############################################################################################################

# READ IN JSON FILE
config_path = '/workspaces/attpc_simulations/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# READ IN & PARSE COMMAND LINE ARGUMENTS
@click.command()
@click.argument('det_file_name')
@click.option('-a', '--avail', is_flag=True, default=False, help='(Optional) Print list of all event numbers available in provided input file.')
@click.option('-e', '--eventselect', default='all', help='(Optional) Choose specific event numbers for analysis, inputted as a string of numbers separated by commas.')
@click.option('-n', '--nucleusselect', default='all', help="(Optional) Choose specific nucleonic constituents to examine.")
@click.option('-v', '--verbose', is_flag=True, default=False, help='(Optional) Toggle individual events/nuclei print statements.')
@click.option('-w','--write', is_flag=True, default=False, help='(Optional) Write data to h5 file.')


def main(det_file_name, avail, eventselect, nucleusselect, verbose, write):

    welcome()
    
    # DEFINE INPUT/OUTPUT PATHS
    eloss_file_name = f'eloss{det_file_name[3:]}'
    input_path = Path(f'/workspaces/attpc_simulations/output/kinematics/{det_file_name}')
    output_path = Path(f'/workspaces/attpc_simulations/output/eloss_calcs/{eloss_file_name}')

    # INITIALIZE DATA STORAGE STRUCTURE
    selected_events = []
    selected_nuclei = []
    event_vertices = []
    event_total_energies = []
    event_KEs = []
    event_velocities = []
    event_proj_ranges = []
    event_angles_lab = []
    event_angles_cm = []
    event_validities = []
    
    with h5py.File(input_path, 'r') as hdf:


        # ~~PRINT INPUT/OUTPUT FILE PATHS
        print('Provided Kinematics File Location:       ', input_path)
        print('Generated Energy Loss File Location:     ', output_path)


        # ~~CREATE DICTIONARY FOR NUCLEUS ORDER, PARALLEL TO ATTPC-ENGINE'S H5 FILE CONVENTION
        nucl_key = ['t', 'p', 'e', 'r']
        nucl = ['TARGET', 'PROJECTILE', 'EJECTILE', 'RESIDUAL']
        nucl_dict = {k: v for k, v in zip(nucl_key, nucl)}
        def get_nucl(key):
            return nucl_dict.get(key, 'Key not found.')
        

        # ~~PRINT ALL EVENT NUMBERS AVAILABLE IN FILE IF REQUESTED
        if avail:
            print(f"\nAvailable events from provided kinematics file '{det_file_name}':\n")
            event_nums = get_eventnums(hdf)

            

        # ~~STORE ALL ISOTOPIC NOTATION IN A LIST
        rxn = config['rxn_info']
        Zs = [rxn['Z_projlike'], rxn['Z_tarlike'], rxn['Z_ejectile'], rxn['Z_residual']]
        As = [rxn['A_projlike'], rxn['A_tarlike'], rxn['A_ejectile'], rxn['A_residual']]
        proj_latex = nuclear_map.get_data(Zs[0], As[0]).isotopic_symbol
        tar_latex =  nuclear_map.get_data(Zs[1], As[1]).isotopic_symbol
        eje_latex =  nuclear_map.get_data(Zs[2], As[2]).isotopic_symbol
        res_latex =  nuclear_map.get_data(Zs[3], As[3]).isotopic_symbol 
        nucl_latex = [proj_latex, tar_latex, eje_latex, res_latex]
        
        print(f'\nREACTION OF INTEREST:')
        if rxn['inverse_kinematics']:
            print(f'          {tar_latex}({proj_latex},{eje_latex}){res_latex}')
        else:
            print(f'          {proj_latex}({tar_latex},{eje_latex}){res_latex}')
        
        print(f"\nKinematic Direction:     {('Inverse Kinematics' if rxn['inverse_kinematics'] else 'Forward Kinematics')}")


        # STORE ANY SPECIFIC EVENT NUMBERS OF INTEREST IF REQUESTED
        if eventselect.strip().lower() != 'all':
            selected_events = [f'{num.strip()}' for num in eventselect.split(',') if num.strip().isdigit()]
            print(f'\nSelected Events:         {selected_events}')


        # STORE ANY SPECIFIC NUCLEI OF INTEREST IF REQUESTED
        if nucleusselect == 'all':
            selected_nuclei = [['TARGET',0], ['PROJECTILE',1], ['EJECTILE',2], ['RESIDUAL',3]]
        else: 
            user_nucl_input = [f'{nuc.strip()}' for nuc in nucleusselect.replace(' ', '').split(',')]
            for i in nucl_key:
                index_ = nucl_key.index(i)
                if i in user_nucl_input:
                    selected_nuclei.append([get_nucl(i),index_])
        print(f'Selected Nuclei:         {[i[0] for i in selected_nuclei]}\n')


        # DEFINE TARGET MATERIAL
        targ_config = config['target']
        print(f"Target:       {config['target']['compound_name']}")
        tar_data = spytar.TargetData(
            compound = targ_config['compound'],
            pressure = targ_config['pressure'],
            thickness = targ_config['thickness']
        )
        targ = spytar.GasTarget(tar_data, spynuc.NuclearDataMap())


        # DEFINE PROJECTILE 
        proj = nuclear_map.get_data(rxn['Z_projlike'],  rxn['A_projlike'])
        print(f"Projectile:   {proj_latex}\n")


        # CYCLE THROUGH ANALYSIS FOR EACH EVENT
        for event_name in selected_events: 
            event_data = hdf[f'/data/event_{event_name}'] 
        
            cprint(f"/////////////////////////////////   EVENT #{event_name}   ///////////////////////////////////// \n",verbose)

            # Extract  attributes from file
            group = hdf['/data']
            Z = group.attrs['proton_numbers']
            A = group.attrs['mass_numbers']
            cprint(f'Nuclear Constituents:     {nucl_latex}',verbose)

            # Extract vertex coordinates (location of simulated reaction event)
            x0 = event_data.attrs['vertex_x']
            y0 = event_data.attrs['vertex_y']
            z0 = event_data.attrs['vertex_z']
            event_vertices.append([x0,y0,z0])
            X_0 = np.array([x0,y0,z0])
            cprint(f'Reaction Vertex:          ({x0:.4f},{y0:.4f},{z0:.4f})',verbose)
            cprint(f'\nNuclear Constituent info:',verbose)

            # Extract four-momentum components for each nucleus
            Px = event_data[0]
            Py = event_data[1]
            Pz = event_data[2]
            P = np.array([Px,Py,Pz])
            E  = event_data[3]
            event_total_energies.append(E)
            cprint('\n  Relativistic Energies, in MeV:',verbose)
            for n in selected_nuclei:
                cprint(f"       {n[0]}:  {E[n[1]]:.2f}",verbose)


            # Calculate energy range and conduct event visibility test for each nuclear constituent:
            nucl_KEs = []
            nucl_velocities = []
            nucl_proj_ranges = []
            nucl_angles_lab = []
            nucl_angles_cm = []
            nucl_validities = []

            for i in selected_nuclei:
                cprint("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n",verbose)
                cprint(f"          NUCLEUS: {nucl_latex[i[1]]} {i[0]} \n",verbose)
        
                energy_i = getKE(Z[i[1]],A[i[1]],E[i[1]])
                nucl_KEs.append(energy_i)

                P_i = np.array([round((P[k][(i[1])]),3) for k in range(len(P))])
                cprint(f"3-Momentum:\n      {P_i}\n",verbose)

                V = getV(np.array([P[j][i[1]] for j in range(len(P))]),E[i[1]])
                nucl_velocities.append(V)
                cprint(f"Velocity:\n      {V}\n",verbose)
        
                range_ = targ.get_range(proj, energy_i)  # calculate results for projectile with energy i
                nucl_proj_ranges.append(range_)
                cprint(f"Projectile Range:\n      {nucl_proj_ranges[i[1]]}\n",verbose)

                try:
                    is_valid = test_event_validity(X_0, V, range_,config["detector_geometry"]["R"],config["detector_geometry"]["H"],verbose)
                    nucl_validities.append(is_valid)
                except ValueError as e:
                    nucl_validities.append(False)

                cprint("\n\nEvent Validity:\n",verbose)
                cprint(f"   Angle:  {nucl_validities[i[1]][0]}\n",verbose)
                cprint(f"   Intersection Point:  {nucl_validities[i[1]][1]}\n",verbose)
                cprint(f"   Detected?  {nucl_validities[i[1]][2]}",verbose)

            event_KEs.append(nucl_KEs[selected_events.index(event_name)])
            event_total_energies.append(nucl_total_energies)
    list_lengths = [
    len(selected_events),
    len(event_total_energies),
    len(event_KEs),
    len(event_velocities),
    len(event_proj_ranges),
    len(event_angles_lab),
    len(event_angles_cm),
    len(event_validities),
    len(nucl_latex),
    len(event_vertices)
]
    print(list_lengths)

    
    # WRITE RESULTS TO H5 FILE
    run_num = det_file_name[4:8] 
    if write: 
        writer = dataWriter(filename=output_path,run_num=run_num)
        for i in range(len(selected_events)):
            event = Event(
                event_num = selected_events[i],
                event_total_energies = event_total_energies[i],
                event_KEs = event_KEs[i],
                event_velocities = event_velocities[i],
                event_proj_ranges = event_proj_ranges[i],
                event_angles_lab = event_angles_lab[i],
                event_angles_cm = event_angles_cm[i],
                event_validities = event_validities[i],
                nucl_latex = nucl_latex[i],            
                event_vertex = event_vertices[i]
            )
            writer.add_event(event)
        writer.write_data()

if __name__ == '__main__':
    main()