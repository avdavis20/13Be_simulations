import h5py # type: ignore
from dataclasses import dataclass, field #type: ignore
from typing import List #type: ignore

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

# create real class (def init)
# point cloud example for class creation
# spyral writer

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


