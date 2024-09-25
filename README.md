# 13Be_simulations
Various simulations for 12Be(d,p)13Be thesis experiment to be held at RCNP.

# attpc_simulations

SIMULATION PIPELINE: 

0. Input experimental parameters into config file titled "kinesim_config.json".
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





To do:
- Make sense of det output file

add chunk of code before writer that gives raw detector output, or just write a different writer
create a data frame for raw detector data: 
    raw detector data -> 

point cloud is 3 elements
(pad id num, electrons recorded, and time)

writing system  takes 3 elements and turns it into a specific style

so detector output can be put in detector analysis




- attpc engine writer class of detector module
        spyralwriter = takes engine data by events

- write to disk throughout process, not just at the end
