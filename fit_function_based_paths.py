import opensim as osim
import numpy as np
from build_hopper import build_hopper
from time import time
from plotting import plot_coordinate_samples, plot_path_lengths, plot_moment_arms

# Create an OpenSim model of a "hopper" system.
hopper = build_hopper(point_on_line_constraint=False)
state = hopper.initSystem()

# Fit a set of function-based paths to the hopper model
fitter = osim.PolynomialPathFitter()
fitter.setModel(osim.ModelProcessor(hopper))
fitter.setCoordinateValues(osim.TableProcessor('hopper_states.sto'))

results_dir = 'path_fitting_results'
fitter.setOutputDirectory(results_dir)
fitter.setMaximumPolynomialOrder(5)
fitter.setNumSamplesPerFrame(10)
fitter.setGlobalCoordinateSamplingBounds(osim.Vec2(-30, 30))
fitter.setUseStepwiseRegression(True)
fitter.setPathLengthTolerance(1e-3)
fitter.setMomentArmTolerance(1e-3)
fitter.run()

# Plot the results
plot_coordinate_samples(results_dir, hopper.getName())
plot_path_lengths(results_dir, hopper.getName())
plot_moment_arms(results_dir, hopper.getName())
