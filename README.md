OpenSim Hopper Demo
-------------------

1. Install Anaconda. (I prefer miniconda for a simpler install).
2. Run the following install commands to install the OpenSim Conda package. The third
   line will also install the version of NumPy needed by OpenSim.

> conda create -n hopper_demo python=3.12
> conda activate hopper_demo
> conda install -c opensim-org opensim

3. If you re-run the function-based path fitting script, you will need to install
   matplotlib to plot the results.

> conda install matplotlib

4. Simulate the hopper.

> python simulate_hopper.py

This script will print out the simulation performance, and if using function-based paths,
the expression defining the path length and moment arm relationships.

Visualizing the results (if desired)
------------------------------------
OpenSim has a built-in visualizer, but unfortunately it currently cannot be used with
our Conda packages. However, the results can be visualized in the OpenSim GUI.

1. Download the results SimTK.org: https://simtk.org/projects/opensim.

2. In the GUI, go to "File > OpenSim Model" and load the model which was saved to the 
   "hopper.osim", OpenSim's XML file format for storing models.

3. Next, go to "File > Load Motion" and select the "hopper_states.sto" file (which was 
   also created by the simulation script).

4. Click the play button at the top middle of the GUI to view the motion.
