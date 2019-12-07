##Running the Solver:
1. Install the following python packages/libraries:
	a. Google OR-Tools (*python -m pip install --upgrade --user ortools*)
	b. NetworkX (*pip install --user networkx*)
	c. Scipy
	d. Numpy
		
2. Ensure that you have student_utils.py and utils.py.

3. Run solver.py using the following command given:
	
	```
	python solver.py [--all] input_file output_directory
	```
	Specificy [--all] if want to run on all input files in the input_directory. Else just runs on one specific file.


## Details about the solver:

All of the solver code is contained in the given solver.py file. We implemented 3 solver functions:

	solve() - a "naive" shortest path solver
	solve_tsp() - a solver that uses google OR's TSP approximator
	solve_tsp_grouped() - a solver that optimizes solve_tsp() by clustering vertices for better optimized dropoffs

When running the python solver.py command given above, the solver runs all 3 algorithms and determines the optimal solution out of the 3.
	




	

