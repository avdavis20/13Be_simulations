# DETECTOR APPLICATION TO KINEMATICS
# by Alyssa Davis, 2024

# IMPORT LIBRARIES
from attpc_engine.detector import ( #type: ignore
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
    run_simulation,
    SpyralWriter,
) 
from attpc_engine import nuclear_map #type: ignore
from spyral_utils.nuclear.target import TargetData, GasTarget #type: ignore
from pathlib import Path
import json
import sys


# LOAD CONFIG FILE
config_path = '/workspaces/attpc_simulations/config.json'
with open(config_path, 'r') as f:
    conf = json.load(f)

# READ/PARSE IN-LINE ARGUMENTS
sim_file_name = sys.argv[1]
det_file_name = f"run{sim_file_name[3:]}"

# DEFINE INPUT/OUTPUT PATHS
paths = conf['workspace']
input_path = Path(f"{paths['kin_output_path']}/{sim_file_name}")
output_path = Path(f"{paths['det_output_path']}/")

# LOAD TARGET INFO
targ_conf = conf['target']
print('TARGET MATERIAL: ', targ_conf['compound_name'])
targ_data = TargetData(
    compound = targ_conf['compound'],
    pressure = targ_conf['pressure'],
    thickness = targ_conf['thickness']
)

# DECLARE DETECTOR
gas = GasTarget(targ_data, nuclear_map)

# DEFINE DETECTOR, ELECTRONICS, AND PAD PARAMETERS
detector = DetectorParams(
    length=1.0,
    efield=45000.0,
    bfield=2.85,
    mpgd_gain=175000,
    gas_target=gas,
    diffusion=0.277,
    fano_factor=0.2,
    w_value=34.0,
)

electronics = ElectronicsParams(
    clock_freq=6.25,
    amp_gain=900,
    shaping_time=1000,
    micromegas_edge=10,
    windows_edge=560,
    adc_threshold=40.0,
)

pads = PadParams()


# CONFIGURE SETTINGS, RUN AND WRITE SIMULATION TO H5 FILE
config = Config(detector, electronics, pads)
writer = SpyralWriter(output_path, config)

def main():
    run_simulation(
        config,
        input_path,
        writer,
    )

if __name__ == "__main__":
    main()