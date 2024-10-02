import numpy as np
import vector
c = 1.0 # natural units


# SIMULATION OF GIVEN INFO

def calc_angles(P_lab):
    """ Calculates azimuthal and polar angles for each particle in a reaction event from both lab and center of mass frames.
    
    Params: 
    - P [array]: array of 4-vectors describing each particle's motion in an event. 

    Returns: 
    - angles_lab [float]: polar and azimuthal angles in lab frame
    - angles_cm [float]: polar and azimuthal angles in cm frame
    """

    # Calculate initial total 4-momentum in the lab frame
    Ptot_lab = (P_lab[0]+P_lab[1])
    print(P_lab[0])

    # Boost total 4-momentum array into center of mass frame
    P_cm: vector.MomentumObject4D = P_lab.boostCM_of(Ptot_lab)

    # Calculate polar/azimuthal angles of particle trajectories in both frames.
    angles_lab = np.array([P_lab.rho, P_lab.phi])
    angles_cm = np.array([P_cm.rho, P_lab.phi])
    print(angles_lab)
    print(angles_cm)
    
    return angles_lab, angles_cm

# Testing function with fake data to represent 1 reaction event
P_lab = vector.array({
    "px": [1.,2.,3.,4.],
    "py": [5.,6.,7.,8.],
    "pz": [9.,10.,11.,12.],
    "E": [111.,222.,333.,444.]
})
calc_angles(P_lab)
