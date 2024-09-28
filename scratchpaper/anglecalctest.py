import numpy as np

c = 1.0 # speed of light in m/s


# SIMULATION OF GIVEN INFO

def calc_angle(P,E):
    """ Calculates azimuthal angles for each particle in a reaction, in both lab and center of mass frames.
    
    Parameters: 
    - P [array] = 4-element np array of 3-momenta for each particle, measured in the lab frame (MeV/c).
    - E [array] = 4-element np array of total relativistic energies for each particle (including rest mass) (MeV).

    Returns: 
    - vcm [float] = velocity of the center of mass. 
    - gamma (float) = lorentz boost scalar value.
    - Pcm [array] = 4-element np array of 3-momenta for each particle in the CM frame.
    - polar_lab [array] = 4-element np array of polar angles, measured in lab frame.
    - azi_lab [array] = 4-element np array of azimuthal angles, measured in lab frame.
    - polar_cm [array] = 4-element np array of polar angles, measured in CM frame.
    - azi_cm [array] = 4-element np array of azimuthal angles, measured in CM frame. 
    """

    P_norm = [np.sqrt(P[i].dot(P[i])) for i in range(len(P))]

    # Calculate initial total energy in lab frame
    Etot_lab = E[0]+E[1] 

    # Calculate initial total momentum in lab frame
    Ptot_lab = P[0]+P[1]
    Ptot_lab_norm = np.sqrt(Ptot_lab.dot(Ptot_lab))

    # Calculate center of mass velocity vcm
    vcm = ((Ptot_lab*(c**2))/Etot_lab)
    vcm_norm = np.sqrt(vcm.dot(vcm))

    # Calculate the Lorentz factor gamma
    gamma = (1/(np.sqrt(1-((vcm_norm/c)**2))))

    # Calculate momenta in CM frame
    Pcm = np.array([P[i] + (((gamma-1)*vcm.dot(P[i])/(vcm.dot(vcm)))-(gamma*E[i])/(c**2))*vcm for i in range(len(P[0]))])
    Pcm_norm = [np.sqrt(Pcm.dot(Pcm)) for i in range(len(P))]

    # Calculate polar angle in lab frame
    polar_lab = np.array([np.rad2deg(np.arccos(P[i][2]/(P_norm[i]))) for i in range(len(P[0]))])

    # Calculate azimuthal angle in lab frame
    azi_lab = np.array([np.rad2deg(np.arctan2(P[i][1],P[i][0])) for i in range(len(P[0]))])

    # Calculate polar angle in CM frame
    polar_cm = np.array([np.rad2deg(np.arccos(Pcm[i][2]/Pcm_norm)) for i in range(len(P[0]))])

    # Calculate azimuthal angle in CM frame
    azi_cm = np.array([np.rad2deg(np.arctan2(Pcm[i][1],Pcm[i][0])) for i in range(len(P[0]))])

    return vcm, gamma, Pcm, polar_lab, azi_lab, polar_cm, azi_cm

# Testing function with fake data to represent 1 reaction event
P = np.array([[1,1,1],[2,1,1],[3,1,2],[1,1,1]])
E = np.array([100000,900000,800000,700000])
calc_angle(P,E)
