import h5py # type: ignore
from dataclasses import dataclass, field #type: ignore
from typing import List #type: ignore
import sys
import numpy as np
import vector

@dataclass
class Event: 
    event_num: int
    event_validities: List[bool]
    event_total_energies: List[float]
    event_KEs: List[float]
    event_velocities: List[float]
    event_proj_ranges: List[float]
    event_angles_lab: List[float]
    event_angles_cm: List[float]

class dataWriter:
    def __init__(self, filename: str,run_num: int):
        self.filename = filename
        self.run_num = run_num
        self.events= [] # store every event we create in this list
    
    def add_event(self, event: Event):
        self.events.append(event)

    def write_data(self):
        try: 
            with h5py.File(self.filename, 'w') as f:
                run_group = f.create_group(f'{self.run_num}')
                
                for event in self.events:
                    event_group = run_group.create_group(f'event_{event.event_num}')

                    event_group.create_dataset('event_total_energies',data=event.event_total_energies)
                    event_group.create_dataset('event_KEs', data=event.event_KEs)
                    event_group.create_dataset('event_velocities', data=event.event_velocities)
                    event_group.create_dataset('event_proj_ranges', data=event.event_proj_ranges)
                    event_group.create_dataset('event_angles_lab', data=event.event_angles_lab)
                    event_group.create_dataset('event_angles_cm', data=event.event_angles_cm)
                    event_group.create_dataset('event_validities', data=event.event_validities)

                    event_group.attrs['nucl_latex'] = event.nucl_latex
                    event_group.attrs['event_vertex'] = event.event_vertex

            print(f"Data has been successfully written to '{self.filename}'.")

        except ValueError as ve: 
            print(f"A ValueError occurred: {ve}")
        except Exception as e:
            print(f"An error occurred: {e}")
 

class Analyzer():
    def __init__(self,init_info,v=False):
        self.latex = init_info['latex']
        self.nucl_dict = init_info['nucl_dict']
        self.in_path = init_info['in_path']
        self.out_path = init_info['out_path']
        self.v = v
        self.selected_nuclei = []
        self.selected_events = []
        self.nucl_index = []

    def welcome(self):
        print('\nWELCOME TO THE ENERGY LOSS CALCULATOR.\n')
        if self.v == False:
            print("\n...")
            print("Program will perform silent analysis. For annotations, rerun program with '-v' option enabled.")
 
    def config_options(self, config_settings):
        # Unpack config settings...
        avail = config_settings['avail']
        eventselect = config_settings['eventselect']
        nuclselect = config_settings['nuclselect']
        data = config_settings['data']
        write = config_settings['write']
        filename = config_settings['filename']
        run_num = config_settings['run_num']
        
        # Display available event numbers if requested...
        all_events = [key.split('_')[-1] for key in data]
        if avail==True: 
            print(f"Available events numbers: {all_events}")
            sys.exit("Program now closing. Rerun program with -a disabled to preform analysis.")
        
        # Configure selected events...
        if eventselect.strip().lower() != 'all':
            selected_events = [f'{num.strip()}' for num in eventselect.split(',') if num.strip().isdigit()]
        else:
            selected_events = all_events

        # Configure selected nuclei...
        nucl_input = [f'{nuc.strip()}' for nuc in nuclselect.replace(' ', '').split(',')]
        self.nucl_index = [list(self.nucl_dict.keys()).index(i) for i in self.nucl_dict if i in nucl_input]
        selected_nuclei = [list(self.nucl_dict.items())[i][1] for i in self.nucl_index]

        # configure dataWriter...
        if write: 
            dataWriter(filename,run_num)
        
        return selected_events, selected_nuclei

    def print_rxn_info(self, rxn_info):
        # Unpack rxn info...
        self.events = rxn_info['events']
        self.nuclei = rxn_info['nuclei']
        kindir = rxn_info['kindir']

        # Print statements
        if self.v:
            print(f'Provided Kinematics File Location:\n{self.in_path}\n')
            print(f'Generated Energy Loss File Location:\n{self.out_path}') 
            print(f'\nSelected Events:    {self.events}')
            print(f'Selected Nuclei:    {self.nuclei}')
            print(f'\nREACTION:       {self.latex[1]}({self.latex[0]},{self.latex[2]}){self.latex[3]}')
            print(f"\nTarget:         {self.latex[0]}")
            print(f"Projectile:     {self.latex[1]}")
            print(f"\nKinematics:     {('Inverse' if kindir else 'Forward')}")

    def get_event_info(self,event_info):
        # unpack event info...
        event_name = event_info['event_name']
        data = event_info['event_data']
        X0 = np.array([data.attrs['vertex_x'],data.attrs['vertex_y'],data.attrs['vertex_z']])
        P_lab: vector.MomentumObject4D = vector.array({
                "px": data[0],
                "py": data[1],
                "pz": data[2],
                "E": data[3]
            })
        
        #print statements
        if self.v:
            print(f"/////////////////////////////////   EVENT #{event_name}   ///////////////////////////////////// \n")
            print(f'Nuclear Constituents:     {self.latex}')
            print(f'Reaction Vertex:          {X0}')
            print(f'\nNuclear Constituent info:')
            print('\n  Relativistic Energies, in MeV:')
            for n in self.nuclei:
                print(f"       {n}:  {P_lab['E'][self.nuclei.index(n)]:.2f}")

        return X0,P_lab

