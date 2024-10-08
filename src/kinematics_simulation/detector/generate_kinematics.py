# GENERATE KINEMATIC SIMULATION

# IMPORT LIBRARIES
from attpc_engine.kinematics import ( # type: ignore
    KinematicsPipeline,
    KinematicsTargetMaterial,
    ExcitationGaussian,
    PolarUniform,
    run_kinematics_pipeline,
    Reaction,
) 
from attpc_engine import nuclear_map # type: ignore
from spyral_utils.nuclear.target import TargetData, GasTarget # type: ignore
from spyral_utils.nuclear import NuclearDataMap # type: ignore
from pathlib import Path
import json
import numpy as np

# READ CONFIG FILE
config_path = Path("/Users/Owner/Desktop/AT-TPC Research/13Be_simulations/config.json")
with open(config_path, 'r') as f:
    config = json.load(f)
    print("Loading config file.....")

# DEFINE FILE NAME AND OUTPUT PATH
sim_file_name = 'sim_0001.h5'
output_path = Path(f"{config['workspace']['kin_output_path']}/{sim_file_name}")

# LOAD TARGET MATERIAL
targ_config = config['target']
targ_data = TargetData(
    compound = targ_config['compound'],
    pressure = targ_config['pressure'],
    thickness = targ_config['thickness']
)
targ = GasTarget(targ_data, NuclearDataMap())
print(f"TARGET MATERIAL: {targ_config['compound_name']}")

nevents = 10000
beam_energy = config['beam_energy'] # MeV

print(f"Beam Energy: {beam_energy} MeV")

rxn = config['rxn_info']

pipeline = KinematicsPipeline(
    [
        Reaction(
            target=nuclear_map.get_data(rxn['Z_targ'], rxn['A_targ']), # deuteron
            projectile=nuclear_map.get_data(rxn['Z_proj'], rxn['A_proj']), # 12Be
            ejectile=nuclear_map.get_data(rxn['Z_ejec'], rxn['A_ejec']), # proton
        )
    ],
    [ExcitationGaussian(0.0, 0.001)], # No width to ground state
    [PolarUniform(0.0, np.pi)], # Full angular range 0 deg to 180 deg
    beam_energy, # MeV
    target_material=KinematicsTargetMaterial(
        material=targ, z_range=(0.0, 1.0), rho_sigma=0.007
    )
)

# RUN KINEMATICS PIPELINE
def main():
    run_kinematics_pipeline(pipeline, nevents, output_path)

if __name__ == "__main__":
    main()