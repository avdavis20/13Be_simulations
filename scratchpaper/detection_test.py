import numpy as np
import vector


c_const = 1      # natural units
R = 11          # radius of detector
H = 6           # height of detector


# ANALYZE TRAJECTORY OF ONE PARTICLE
def detection_test(X0,P,range_):
    """
    Parameterizes the trajectory of particle and applies geometric detector 
    constraints. Determines whether or not particle will be detected.

    Params:

    -- X : 3-element array of starting position of particle.
    -- P : 4-element vector array object of particles lab frame 3-momentum and energy.
    -- range_ : particle's stopping distance within detector medium.
    

    Returns: 

    --

    """

    # FIRST, DETERMINE IF PARTICLE WILL BE DETECTED 
    # Note: all of this step is done in the LAB frame.

    # Split 4-momentum vector into 3-momentum and energy
    P_lab = P[0:3] # particle's 3-momentum, in MeV/c
    E_lab = P[3] # particle's energy, in MeV
    [x0,y0,z0] = X0 # particle's starting location

    # Calculate the particle's velocity in the lab frame
    print(P_lab)
    print((P_lab * c_const**2) / (E_lab) )
    V_lab = [vx,vy,vz] = (P_lab * c_const**2) / (E_lab) 

    # Calculate roots of perameterized equations of motion with constraint
    a = vx**2 + vy**2
    print(a)
    print(vx)
    b = 2 * (vx + vy)
    c = x0**2 + y0**2 - R**2

    # Calculate discriminant
    disc = b**2 - (4*a*c)

    # Find roots of polynomial
    if a == 0:  # If the particle never intersects the detector wall...
        t1,t2 = ((-z0)/vz), ((H-z0),vz) # find intersection time with detector caps.
    else: 
        t1,t2 = (-b + (np.sqrt(disc) / (2.0 * a))), (-b - (np.sqrt(disc) / (2.0 * a)))

    # Remove the t that is negative, as its parametric trajectory was in the wrong direction
    t_values = [t for t in [t1, t2] if t > 0]
    if not t_values:
        raise ValueError("Geometrical Constraint Error: No positive t-values found, indicating no valid intersection in the trajectory direction.")
    t = min(t_values)

    # Calculate intersection point
    x_intersect = x0 + vx * t
    y_intersect = y0 + vy * t
    z_intersect = z0 + vz * t    

    # Calculate minimized geometric distance btwn X0 and the constraint boundaries
    geo_dist = geo_dist = np.sqrt((x_intersect - x0)**2 + (y_intersect - y0)**2 + (z_intersect - z0)**2)

    # Determine if particle will be detected based on geometric allowance of range distance
    if geo_dist > range_:
        is_detected = True
    else: 
        is_detected = False

    print(f'Reaction Vertex:     {X0}')
    print(f'4-momentum:          {P}')
    print(f'3-momentum:          {P_lab}')
    print(f'Energy:              {E_lab}')
    print(f'Lab velocity:        {V_lab}')
    print(f'a, b, c:             {a,b,c}')
    print(f'Particle detected?   {is_detected}')
    
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