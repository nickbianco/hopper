import os
import opensim as osim
import numpy as np
from build_hopper import build_hopper
from time import time

use_function_based_paths = True

# Create an OpenSim model of a "hopper" system. 
hopper = build_hopper()

if use_function_based_paths:
    # Use a ModelProcessor to replace the original muscle geometry with 
    # function-based paths.
    modelProcessor = osim.ModelProcessor(build_hopper())
    modelProcessor.append(osim.ModOpReplacePathsWithFunctionBasedPaths(
        os.path.join('path_fitting_results', 'hopper_FunctionBasedPathSet.xml')))
    hopper = modelProcessor.process()

    # The function-based path for the vastus muscle is only a function of the knee
    # coordinate, and in this case, it is linear. Print out the coefficients.
    vastus = hopper.getMuscles().get('vastus')
    path = osim.FunctionBasedPath.safeDownCast(vastus.getPath())
    lengthFunc = osim.MultivariatePolynomialFunction.safeDownCast(
            path.getLengthFunction()) 
    
    import pdb; pdb.set_trace()
    coeffs = lengthFunc.getCoefficients()

    # Moment arms are calculated as the partial derivative of the length function with 
    # respect to the dependent coordinates. Therefore, the moment arm function for the
    # hopper is just a constant value, negated based on OpenSim's sign convention for 
    # moment arms. The lengthening speed function is the time derivative of the length
    # function, which is the product of the moment arm and the generalized speed (with 
    # another negative sign to account for the negation of the moment arm).
    print(f'\nFunction-based path information for the vastus muscle')
    print( '-----------------------------------------------------' )
    print(f'Length function:             l(q) = {coeffs[0]:2f} + {coeffs[1]:2f}*q_knee')
    print(f'Moment arm function:         r(q) = -dl/dq = -{coeffs[1]:2f}')
    print(f'Lenghthening speed function: ldot(q) = dl/dt = -r(q)*qdot = {coeffs[1]:2f}*qdot_knee')
    print( '-----------------------------------------------------\n' )

# Initialize the system and print the model to an XML file.
state = hopper.initSystem()
hopper.printToXML('hopper.osim')

# Create a Manager to simulate the model. The Manager class contains an integrator and 
# and time-stepping class to manage the simulation. The default integrator is a 
# Runge-Kutta 4th order method with adaptive step size control.
manager = osim.Manager(hopper)
manager.initialize(state)

# Simulate the model for 5 seconds and measure the elapsed time.
start_time = time()
manager.integrate(5.0)
end_time = time()
elapsed_time = (end_time - start_time) * 1000 
real_time_factor = 5000.0 / elapsed_time  # 5000 ms for 5 seconds of simulation
print(f'Hopper simulation completed in {elapsed_time:.2f} milliseconds '
      f'({real_time_factor:.2f} faster than real time).')

# Save the states table to a file.
statesTable = manager.getStatesTable()
sto = osim.STOFileAdapter()
sto.write(statesTable, 'hopper_states.sto')
