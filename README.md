# 13Be_simulations
Various simulations for 12Be(d,p)13Be thesis experiment to be held at RCNP.

# attpc_simulations

SIMULATION PIPELINE: 

0. Input simulation parameters into config file titled "config.json".
1. Generate kinematics using script titled "generate_kinematics.py".
2. Apply detector geometry to kinematic results using script titled "apply_detector.py".
3. Run eloss calculations using script titled "eloss_calc.py".


HOW TO RUN PROGRAMS:

Kinematics: python generate_kinematics.py {simulation_file_name}
Detector: python apply_detector.py {simulation_file_name}
Eloss: python eloss_calc.py {simulation_file_name}


EDUCATIONAL REFERENCES FOR ATTPC-SPECIFIC PYTHON PACKAGES USED:
- ATTPC_ENGINE DOCUMENTATION: https://attpc.github.io/attpc_engine/
- SPYRAL-UTILS DOCUMENTATION: https://attpc.github.io/spyral-utils/



How I want this repo to be structured (I think?):
13Be_simulations
-- .venv
-- .gitignore
-- README.md
-- src
    -- __init__.py
    -- generate_kinematics.py
    -- apply_detector.py
    -- eloss_calc.py
-- packages
    -- data_writer.py
    -- converth5.py
-- output
    -- detector (where apply_detector data is written to)
    -- kinematics (where generate_kinematics data is written to)
    -- eloss_calcs (where eloss_calc data is written to)
-- notebooks
    -- graph_results.ipynb
-- config.json
-- requirements.txt
    

