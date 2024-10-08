import numpy as np
import vector
c = 1.0 # natural units


# CALCULATE ANGLE 
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
    
    print('ANGLE CALCULATION RESULTS...\n\n')
    print(f'Lab Frame 4-vectors: \n{P_lab}\n')
    print(f'CM Frame 4-vectors: \n{P_cm}\n')
    print(f'LAB ANGLES:\n   Polar: {angles_lab[0]}\n   Azimuthal: {angles_lab[1]}\n')
    print(f'CM ANGLES:\n   Polar: {angles_cm[0]}\n   Azimuthal: {angles_cm[1]}')

    return angles_lab, angles_cm

# Testing function with fake data to represent 1 reaction event
P_lab = vector.array({
    "px": [1.,2.,3.,4.],
    "py": [5.,6.,7.,8.],
    "pz": [9.,10.,11.,12.],
    "E": [111.,222.,333.,444.]
})
calc_angles(P_lab)
