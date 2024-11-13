import spyral_utils.nuclear.target as spytar 
import spyral_utils.nuclear as spynuc 
from utils.writer_utils import dataWriter
from attpc_engine import nuclear_map 
from pathlib import Path
import sys
import numpy as np
import vector
import json

class elossUtils():
    def __init__(self):
        # Initialize Path Variables/File Names
        self.in_dir = None
        self.out_dir = None
        self.in_filename = None
        self.out_filename = None
        self.config_path = None

        # Initialize Config data
        self.config = None   

        # Initialize constants and h5 file storage conventions
        self.nucl_dict = {'t': 'TARGET', 'p': 'PROJECTILE', 'e': 'EJECTILE', 'r': 'RESIDUAL'}
        self.c = 1

        # Initialize Analysis Scope     
        self.nucl_index = None              
        self.selected_events = None
        self.selected_nuclei = None

        # Initialize Write, Verbose, and Debug Flags
        self.write = None
        self.verbose = None
        self.debugDetect = None
        self.debugAngles = None
        self.debugWrite = None
        
        # Initialize Isotope/Rxn Info
        self.Z = None
        self.A = None
        self.kindir = None
        self.latex = None

        # Initialize Detector Geometry Info
        self.R = None
        self.H = None
        self.dead_zone = None

        # Initialize Event Data Storage
        self.P_lab = None
        self.X0 = None
        self.V_lab = None
        self.event_KEs = None
        self.angles_lab = None
        self.angles_cm = None

        # Initialize Writer 
        self.writer = None


    def get_config_data(self,config_path):
        with open(config_path, 'r') as f: self.config = json.load(f)    # load config file

        self.in_dir = Path(self.config['workspace']['kin_output_path'])            # input dir
        self.out_dir = Path(self.config['workspace']['eloss_output_path'])         # output dir

        self.Z = np.array([list(self.config['rxn_info'].items())[i][1] for i in range(1,8,2)]) # iso proton numbers
        self.A = np.array([list(self.config['rxn_info'].items())[i][1] for i in range(2,9,2)]) # iso mass numbers
        
        self.latex = [nuclear_map.get_data(self.Z[i],self.A[i]).isotopic_symbol for i in range(0,4)]  # iso latex format
        self.kindir = self.config['rxn_info']['inverse_kinematics']  # kinematic direction of rxn

        targ_data = spytar.TargetData(
            compound = self.config['target']['compound'],
            pressure = self.config['target']['pressure'],
            thickness = self.config['target']['thickness']
        )
        self.targ = spytar.GasTarget(targ_data, spynuc.NuclearDataMap())   # rxn targ definition
        self.proj = nuclear_map.get_data(self.Z[0],  self.A[0])                      # rxn proj definition

        self.R = self.config['detector_geometry']['R']     # radius of detector
        self.H = self.config['detector_geometry']['H']     # height of detector     
        self.dead_zone = self.config['detector_geometry']['dead_zone']  # radius of the opening at end of attpc


    def welcome(self):
        print(
            r"""
------------------------------------------------------

                                               __
                                            ."`  `".
                                           /   /\   \
        WELCOME TO THE ENERGY LOSS        |    \/    |  
                CALCULATOR.                \   ()   /           
                                            '.____.'
                                             {_.="}
                                             {_.="}
                                             `-..-`    

------------------------------------------------------
            """)
        

    def config_options(self, flags):
        # Unpack flags...
        avail = flags['avail']
        eventselect = flags['eventselect']
        nuclselect = flags['nuclselect']
        self.write = flags['write']
        self.verbose = flags['verbose']
        self.debugDetect = flags['debugDetect']
        self.debugAngles = flags['debugAngles']
        self.debugWrite = flags['debugWrite']
        self.in_filename = flags['in_filename']
        self.out_filename = flags['out_filename']
        self.run_num = flags['run_num']
        eventnum_data = flags['eventnum_data']

        if self.verbose == False:
            print("\n...")
            print("Program will perform silent analysis. For annotations, rerun program with '-v' option enabled.\n")
        
        # Display available event numbers if requested...
        all_events = [key.split('_')[-1] for key in eventnum_data]
        if avail==True: 
            print(f"Available events numbers: {all_events}")
            sys.exit("Program now closing. Rerun program with -a disabled to perform analysis.")
        
        # Configure selected events...
        if eventselect.strip().lower() != 'all':
            self.selected_events = [f'{num.strip()}' for num in eventselect.split(',') if num.strip().isdigit()]
        else:
            self.selected_events = all_events

        # Define nucleus scope...
        nucl_input = [f'{nuc.strip()}' for nuc in nuclselect.replace(' ', '').split(',')]
        self.nucl_index = [list(self.nucl_dict.keys()).index(i) for i in self.nucl_dict if i in nucl_input]
        self.selected_nuclei = [list(self.nucl_dict.items())[i][1] for i in self.nucl_index]

        # Configure dataWriter...
        if self.write: 
            self.writer = dataWriter(self.out_dir, self.out_filename, self.run_num, self.verbose, self.debugWrite)
        
        return self.selected_events, self.selected_nuclei


    def print_rxn_info(self):
        # Print statements
        if self.verbose:
            print(f'Provided Kinematics File Location:\n{self.in_dir}\n')
            print(f'Generated Energy Loss File Location:\n{self.out_dir}') 
            print(f'\nSelected Events:    {self.selected_events}')
            print(f'Selected Nuclei:    {self.selected_nuclei}')
            if self.kindir == True: 
                print(f'\nREACTION:       {self.latex[1]}({self.latex[0]},{self.latex[2]}){self.latex[3]}')
                print(f"\nTarget:         {self.latex[0]}")
                print(f"Projectile:     {self.latex[1]}")
                print(f"\nKinematics:     Inverse")
            else: 
                print(f'\nREACTION:       {self.latex[0]}({self.latex[1]},{self.latex[2]}){self.latex[3]}')
                print(f"\nTarget:         {self.latex[1]}")
                print(f"Projectile:     {self.latex[0]}")   
                print(f"\nKinematics:     Forward")


    def get_event_info(self,event_info):
        # Unpack event info
        event_name = event_info['event_name']
        data = event_info['event_data']
        self.X0 = np.array([data.attrs['vertex_x'],data.attrs['vertex_y'],data.attrs['vertex_z']])

        # Define Lab frame four-momenta
        self.P_lab:vector.MomentumObject4D = vector.array({
                "px" : data[:,0],
                "py" : data[:,1],
                "pz" : data[:,2],
                "E"  : data[:,3]
            })
        
        # Print statements
        if self.verbose:
            print(f"\n/////////////////////////////////   EVENT #{event_name}   ///////////////////////////////////// \n")
            print(f'Nuclear Constituents:     {self.latex}')
            print(f'Reaction Vertex:          {self.X0}')
            print(f'\nIsotope info:')
            print(f'\n  Four-momenta: \n\n{self.P_lab}')
            print('\n\n  Relativistic Energies, in MeV:\n')
            for i in range(0,4):
                print(f"       {self.selected_nuclei[i]} {self.latex[i]}:  {self.P_lab['E'][i]:.2f}")
    

    def test_detection(self, X0, P):
        """
        Parameterizes the trajectory of particle and applies geometric detector constraints 
        to determine whether or not particle will be detected.

        Params:
        -- X0 [float]: Event vertex in R3.
        -- P [vector.MomentumNumpy4D]: Vector array of 4-momentum vectors for all 4 particles in event.
    
        Attributes:  
        -- is_detected (Bool): flag indicating if particle was detected.
        """

        # Parametrized particle trajectory:
        # x(t) = x0 + vx * t
        # y(t) = y0 + vy * t
        # z(t) = z0 + vz * t

        # Cylindrical surface boundary constraint:
        # x(t)^2 + y(t)^2 = R^2
        # 0 < z(t) < H

        # Unpack event vertex, a.k.a., particles' starting location
        [x0,y0,z0] = np.array(X0)  

        # Calculate the particles' velocities in the lab frame
        self.V_lab: vector.MomentumObject3D = vector.array({
            'x': np.array(P['px'] / P['E']),
            'y': np.array(P['py'] / P['E']),
            'z': np.array(P['pz'] / P['E'])
        })
        vx, vy, vz = self.V_lab['x'], self.V_lab['y'], self.V_lab['z']
        
        # Initialize data storage for each particle's detection status
        is_detected = [] 

        for i in range(len(vx)):
            if self.debugDetect == True:
                print(f"\n\n     ----- {self.latex[i]} {self.selected_nuclei[i]} NUCLEUS -----\n\n")
                print(f"Initial position: \n   x0 = {x0:.3f},\n   y0 = {y0:.3f},\n   z0 = {z0:.3f}")
                print(f"\nVelocities: \n   vx = {vx[i]:.3f},\n   vy = {vy[i]:.3f},\n   vz = {vz[i]:.3f}\n")

            # Skip particles with zero velocity in all directions
            if vx[i] == 0 and vy[i] == 0 and vz[i] == 0:
                if self.debugDetect == True: 
                    print(f"{self.latex[i]} particle has zero velocity in all directions, skipping.\n")
                is_detected.append(False)
                continue

            # Calculate cap intersections based on vz
            if vz[i] > 0:
                # Particle is moving upward, calculate top cap intersection
                t_top = (self.H - z0) / vz[i]
                t_bot = None  # No intersection with bottom cap
                if self.debugDetect:
                    print(f"Top cap intersection time: \n   t_top = {t_top:.3f} seconds\n")
            elif vz[i] < 0:
                # Particle is moving downward, calculate bottom cap intersection
                t_bot = (- z0) / vz[i]
                t_top = None  # No intersection with top cap
                if self.debugDetect:
                    print(f"Bottom cap intersection time: \n   t_bot = {t_bot:.3f} seconds\n")
            else:
                # No z-motion, so no cap intersections
                t_top, t_bot = None, None
                if self.debugDetect == True: print(f"No z-motion, so no cap intersections.\n")
            
            # Calculate quadratic equation coefficients
            a = vx[i]**2 + vy[i]**2

            if a == 0:
                # No horizontal motion, so no wall intersection
                if self.debugDetect:
                    print(f"No horizontal motion, no wall intersection.\n")
                t_wall = None

            else:
                b = 2* (x0 * vx[i] + y0 * vy[i])
                c = x0**2 + y0**2 - self.R**2

                # Calculate discriminant using coefficient values
                disc = (b**2) - (4 * a * c)

                # Find roots
                if disc < 0: # No intersection with the cylindrical wall
                    t_wall = None
                    if self.debugDetect:
                        print(f"No wall intersection (discriminant is negative).\n")

                else:
                    # Two possible intersection times with the wall
                    t1 = (-b + np.sqrt(disc)) / (2 * a)
                    t2 = (-b - np.sqrt(disc)) / (2 * a)
                    # save smallest of positive t-values
                    t_wall = min([t for t in [t1, t2] if t >= 0], default=None)
                    if self.debugDetect:
                        print(f"Wall intersection times: \n   t1 = {t1:.3f}, \n   t2 = {t2:.3f}, \n   chosen t_wall = {t_wall:.3f}\n")

            # Store all valid intersections in a list so we can find the shortest time
            t_candidates = [ti for ti in [t_top, t_bot, t_wall] if ti is not None and ti >= 0]
            
            if t_candidates:
                t_min = min(t_candidates)
                if self.debugDetect:
                    print(f"Chosen intersection time: \n   t_min = {t_min:.3f} seconds\n")
            else: 
                t_min = None
                if self.debugDetect == True: print(f"No valid intersection times found.\n")

            if t_min is not None:
                # Calculate intersection point based on minimized intersection time parameter
                intersection_pt = vector.array({
                    'x': x0 + vx[i] * t_min,
                    'y': y0 + vy[i] * t_min,
                    'z': z0 + vz[i] * t_min
                })
                if self.debugDetect:
                    print(f"Intersection point:\n   x = {intersection_pt['x']:.3f},\n   y = {intersection_pt['y']:.3f},\n   z = {intersection_pt['z']:.3f}\n")
            
            # Check if the particle passes through the dead zone (hole in the top cap)
            if t_top is not None and t_top == t_min:
                # Calculate radial distance from the center of the top cap
                r_top = np.sqrt(intersection_pt['x']**2 + intersection_pt['y']**2)
                if r_top <= self.dead_zone:
                    if self.debugDetect:
                        print(f"Particle passes through the cap hole (radius {r_top:.3f} <= dead zone {self.dead_zone:.3f}).")
                    is_detected.append(True)
                    continue  # Particle passes through the dead zone, so it's detected
                
                # Calculate linear geometric distance btwn X0 and the constraint boundaries
                geo_dist = np.sqrt((intersection_pt['x'] - x0)**2 + (intersection_pt['y'] - y0)**2 + (intersection_pt['z'] - z0)**2)
                if self.debugDetect:
                    print(f"Geometric distance: \n   {geo_dist:.3f} meters\n")

                # Calculate nuclear range (aka stopping distance) of particle i with energy Ei
                range_ = self.targ.get_range(nuclear_map.get_data(self.Z[i], self.A[i]), self.event_KEs[i])
                if self.debugDetect:
                    print(f"Stopping distance: \n   {range_:.3f} meters\n")

                # Determine if particle will be detected based on geometric allowance of range distance
                detected = geo_dist > range_
                if self.debugDetect == True: print(f"Is {self.latex[i]} particle detected? \n   {'Yes' if detected else 'No'}")
                is_detected.append(detected)
            else:
                # If no intersection, particle is not detected
                is_detected.append(False)
                if self.debugDetect == True: print(f"{self.latex[i]} particle is not detected (no intersection).")

        return is_detected


    def calc_KEs(self, P_lab: vector.MomentumObject4D):
        """ Calculates kinetic energy of particles in a reaction event. 
        
        Params: 
        - P_lab [vector.MomentumNumpy4D]: Vector array of 4-momentum vectors for all 4 particles in event.

        Attributes:
        - event_KEs [float]: Kinetic energy of particles in a given event.
        """
        self.event_KEs = P_lab['E'] - np.array([nuclear_map.get_data(self.Z[i],self.A[i]).mass for i in range(4)])


    def calc_angles(self, P_lab: vector.MomentumObject4D):
        """ Calculates polar and azimuthal angles for each particle of a reaction event,
        from both lab and center of mass frames.

        Params: 
        - P_lab [vector.MomentumNumpy4D]: Vector array of 4-momentum vectors for all 4 particles in event.

        Attributes: 
        - angles_lab [[float],[float]]: polar and azimuthal angles in lab frame (in deg)
        - angles_cm [[float],[float]]: polar and azimuthal angles in cm frame (in deg)
        """

        # Calculate initial total 4-momentum in the lab frame
        Ptot_lab: vector.MomentumObject4D = (P_lab[0] + P_lab[1])

        # Boost total 4-momentum array into center of mass frame
        P_cm: vector.MomentumObject4D = P_lab.boostCM_of(Ptot_lab)

        # Calculate polar/azimuthal angles of particle trajectories in the lab frame
        polar_lab = np.rad2deg(P_lab.theta)
        azi_lab = np.rad2deg(P_lab.phi)
        self.angles_lab = [polar_lab,azi_lab]

        # Calculate polar/azimuthal angles of particle trajectories in the CM frame
        polar_cm = np.rad2deg(P_cm.theta)
        azi_cm = np.rad2deg(P_cm.phi)
        self.angles_cm = [polar_cm,azi_cm]
    
        # Print results
        if self.debugAngles:
            print('\n\n\nANGLE CALCULATION RESULTS...\n\n')
            print(f'Lab Frame 4-vectors: \n{P_lab}\n')
            print(f'CM Frame 4-vectors: \n{P_cm}\n')
            print(f'LAB ANGLES:\n   Polar:     {self.angles_lab[0]}\n   Azimuthal: {self.angles_lab[1]}\n')
            print(f'CM ANGLES:\n    Polar:    {self.angles_cm[0]}\n   Azimuthal: {self.angles_cm[1]}')


