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


EDUCATIONAL REFERENCES ~

- ATTPC_ENGINE DOCUMENTATION: https://attpc.github.io/attpc_engine/
- SPYRAL-UTILS DOCUMENTATION: https://attpc.github.io/spyral-utils/
- GRAPHING RESOURCE: https://how2matplotlib.com/how-to-plot-a-histogram-with-various-variables-in-matplotlib-in-python.html



REPO STRUCTURE ~

13Be_simulations
-- .venv
-- notebooks
    -- __ init __.py
    -- graph_results.ipynb
-- output
    -- detector
    -- kinematics
    -- eloss_calcs
-- scratchpaper
    -- __ init __.py
    -- Put any test code here
-- src
    -- __ init __.py
    -- kinematics_simulation
        -- __ init __.py
        -- detector
            -- __ init __.py
            -- generate_kinematics.py
            -- apply_detector.py
        -- eloss_calcs
            -- __ init __.py
            -- eloss_calc.py
        -- writer
            -- __ init __.py
            -- data_writer.py
-- .gitignore
-- config.json
-- README.md
-- requirements.txt