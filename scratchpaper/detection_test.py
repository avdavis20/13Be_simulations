import numpy as np
import vector


c_const = 1      # natural units
R = 11          # radius of detector
H = 6           # height of detector


# ANALYZE ALL PARTICLE TRAJECTORIES FOR ONE EVENT
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
    print(f'Parameterized intersection point btwn particle trajectory and constraints: \n{t}\n')
    print(f'Particle detected?\n{is_detected}\n\n\n')
    
    return is_detected



X0 = np.array([0.5,0.6,0.7])
range_ = np.array([8.,9.,10.,11.])

# Define vector object array for 4-momentum vectors in lab frame
P_lab = vector.array({
    "px": [1.,2.,3.,4.],
    "py": [5.,6.,7.,8.],
    "pz": [9.,10.,11.,12.],
    "E": [111.,222.,333.,444.]
})

detection_test(X0, P_lab, range_)