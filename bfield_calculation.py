""" 
bfield_calculation
Biot-Savart law for linear filamentary segments
used vecnorm.m  http://bit.ly/3argnkQ

field = bfield(coil,grid)

INPUTS:
coil=[p1,p2,...pn]  pi=[x;y;z] points on the coil
or {coil1,coil2,,...} or {coil1, coil2,...; current1, current2,...}
default: current = 1
grid=[x1,x2,...xm]  xi=[x;y;z] field points

OUTPUTS:
field=[b1,b2,...bm] bi=[bx;by;bz] field calculated at points
in units of mu0*I/4pi [mu0/4pi = 1 G*mm/A]
EXAMPLE:
coil=[1,1,0; -1,1,0; -1,-1,0; 1,-1,0; 1,1,0]';
grid=[0,0,0; 1,1,2]';
bfield(coil,grid)
ans = 
0    0.2092
0    0.2092
5.6569    0.2887
analytic calculation for comparison, see: square_loop.pdf http://bit.ly/2W3AtYL
0,0,8/sqrt(2); 1/2/sqrt(2)-1/4/sqrt(3), 1/2/sqrt(2)-1/4/sqrt(3), 1/2/sqrt(3)]' """

import numpy as np
from numpy.typing import NDArray

# Converts inputs into standardized form. Throws errors if unusable explaining the issue.
def bfield_inputs(coil: NDArray | list, grid: NDArray, currents: NDArray | None = None) -> dict:
	""" Ensures that the parameters passed to the magnetic field calculating function are in the appropriate form. Raises an error if something is off with how the parameters are passed. Returns a dictionary of each of these parameters in a proper form for use in calculating the bfield.

	Parameters
	----------
	coil: NDArray, list
		Multiple coils could be passed as an array if they have the same number of points or as a list of arrays of 3-d points. Can also pass currents in if coil is a list.

		

	grid: NDArray
		Must have size (3) or (n, 3), where n is the number of gridpoints to compute the field at. 

	currents: NDArray, None
		An NDArray of n currents, corresponding to each of the n coils in order.
		
		If coil is a list, currents must be None, as currents would be defined as part of coil. 

		If coil isn't a list and currents is None, defaults to 1 for each coil.

	Returns
	-------
		dict: 
			- "coil" is a list of arrays of discretized coils
			- "grid" is an array of points to measure the magnetic field
			- "current" is an array of currents corresponding to each coil	
	"""
	
	# Ensure that grid is usable for this code
	if (type(grid) != NDArray) and (type(grid) != np.ndarray):
		raise TypeError("grid must be an array of 3-vectors.")
	grid_dims = np.ndim(grid)	
	if grid_dims != 1 and grid_dims != 2:
			raise ValueError("grid as an array should have dimensions 1 or 2.")
	elif grid_dims == 1:
		if len(grid) != 3:
			raise ValueError("grid should be 3-vectors")
		grid = np.array([np.copy(grid)])
	else:
		if len(grid[0]) != 3:
			raise ValueError("grid should be 3-vectors")


	# Ensure that coil is usable for this code
	# If coil is a list containing both coil positions and currents, split into separate arrays
	if (type(coil) == list): # and (type(coil[0]) != list):
		if type(coil[0]) == list: # Coil = [[coil1, coil2, ...], currents]
			if (type(coil[-1]) != NDArray) and (type(coil[-1]) != np.ndarray):
				raise TypeError("coil passed improperly")
			elif currents is not None:
				raise TypeError("Can't pass current values in both coil and currents")
			elif np.ndim(coil[-1]) != 1:
				raise ValueError("Improper dimensions for currents")
			elif len(coil[0]) != len(coil[-1]):
				raise ValueError("Current dimensions must match number of coils")
			currents = np.copy(coil[-1])
			coil = np.array(coil[0])
			
		elif  (type(coil[0]) != NDArray) and (type(coil[0]) != np.ndarray):
			raise TypeError("coil passed improperly")
		
		elif np.ndim(coil[0]) == 3: # Coil = [{coil1, coil2, ...}, currents]
			if currents is not None:
				raise TypeError("Can't pass current values in both coil and currents")
			currents = np.copy(coil[-1])
			coil = np.copy(coil[0])

		elif np.ndim(coil[-1]) == 1: # Coil = [coil, current]
			if currents is not None:
				raise TypeError("Can't pass current values in both coil and currents")
			currents = np.copy(coil[-1])
			coil = np.copy(coil[0])
		
		else: # Coil = [coil1, coil2, ...]
			for coil_i in coil:
				if np.ndim(coil_i) != 2:
					raise ValueError("coil dimensions are incorrect")
			coil = np.array(coil)
		# By this point, current is None if not passed anywhere or an array of currents if passed.
		# Coil is an array (either 2-d or 3-d depending on if multiple coils)


	# Consider possibilities for passing in array for coil. Sets coil_pos as an array with arrays corresponding to coil positions as well as how many coils there are.
	if (type(coil) == NDArray) or (type(coil) == np.ndarray):
		coil_dims = np.ndim(coil) # 2 = single coil, 3 = multiple coils
		if coil_dims != 2 and coil_dims != 3:
			raise ValueError("coil as an array should have dimensions 2 or 3.")
		if coil_dims == 2:
			if len(coil[0]) != 3:
				raise ValueError("Single coil should be 3-vectors")
			coil_pos = np.array([np.copy(coil)])
		else:
			if len(coil[0][0]) != 3:
				raise ValueError("Single coil should be 3-vectors")
			coil_pos = np.copy(coil)
		num_coils = len(coil_pos)
		list_coils = [coil_points for coil_points in coil_pos]

	else: 
		raise TypeError("coil must either be a list or array.")
	

	# Ensure proper passing for currents. Default to 1 if not provided.
	if currents is not None:
		if len(currents) != num_coils:
			raise ValueError("Number of coils must match number of coils.")
	else:
		print("\n Defaulting to current values of 1 for each coil.\n")
		currents = np.ones(num_coils)

	outputs = {
		"coil": list_coils,
		"grid": grid,
		"current": currents
	}
	return outputs






def bfield(coil: list, grid: NDArray, currents: NDArray) -> NDArray:
	""" Computes the magnetic field at each grid point from all coils using a discretized Biot-Savart law.

	Parameters
	----------
	coil: list
		A list of n coils, where each coil is an array of size (m, 3) with m points specifying coil positions.

	grid: NDArray
		An array of size (n, 3) to measure the magnetic field at each of the n points. 

	currents: NDArray, None
		An NDArray of n currents, corresponding to each of the n coils in order.

	Returns
	-------
		NDArray: 
			An array of size (n, 3) of the 3-d magnetic field at each of the points in grid, in order.
	"""
	μ0 = 1.25663706127e-6 # N/A^2
	field = np.zeros_like(grid)			# empty output array of right size
	
	num_coils = len(coil)

	for i in range(len(coil)):
		vRf = grid - coil[i][0] # Vectors from coil point initial to measurement points
		sRf = np.linalg.norm(vRf, axis = 1) # Magnitudes of vectors

		for k in range(1, len(coil[i])): # For each pairing of points in a particular loop

			vRi = np.copy(vRf)
			sRi = np.copy(sRf)
			vRf = grid - coil[i][k] 
			sRf = np.linalg.norm(vRf, axis = 1)
			sRiRf = np.multiply(sRi, sRf)
			
			vRidotvRf = np.sum(np.multiply(vRi, vRf), axis = 1)
			const = (μ0/(4*np.pi))*currents[i]*(sRi + sRf)/(sRiRf*(sRiRf + vRidotvRf))
			# const = (μ0/4π)*I*(|Ri| + |Rf|)/(|Ri||Rf|(|Ri||Rf| + vRi x vRf))
			temp_field = np.cross(vRi, vRf)*const[:, None]
	
			
			field = field + temp_field
			
			



	# for k = 2:size(coil{1,sc},2):	# calculate vector of fields in loop over coil segments
	# 	vRi = vRf; sRi=sRf;		# previous final point = new initial point
	# 	vRf = grid - coil{1,sc}(:,k); # calculate new relative vectors
	# 	sRf = sqrt(sum(vRf.*vRf));	# vecnorm(vRf);
	# 	tmp = sRi.*sRf;
	# 	sfield = sfield + cross(vRi,vRf).*(sRi+sRf)./tmp./(tmp + dot(vRi,vRf));		 # Biot-Savart law, see Eq. 9, http://bit.ly/2DsXUTK
	# if size(coil,1)>1 sfield=sfield*coil{2,sc}; end;   #current in second row
	# 	field = field + sfield;
	return field


def build_M_bfield(coil: list, grid: NDArray, currents: NDArray) -> NDArray:
	""" Computes the magnetic field at each grid point from all coils using a discretized Biot-Savart law.

	Parameters
	----------
	coil: list
		A list of n coils, where each coil is an array of size (m, 3) with m points specifying coil positions.

	grid: NDArray
		An array of size (n, 3) to measure the magnetic field at each of the n points. 

	currents: NDArray, None
		An NDArray of n currents, corresponding to each of the n coils in order.

	Returns
	-------
		dict: 
			{"Bx": NDArray, "By": NDArray, "Bz": NDArray}
			
	"""
	μ0 = 1.25663706127e-6 # N/A^2
	num_coils = len(coil)
	num_points = len(grid)

	field_x = np.zeros((num_points, num_coils))		# empty output array of right size
	field_y = np.copy(field_x)
	field_z = np.copy(field_x)
	
	

	for i in range(len(coil)):
		coil_field = np.zeros_like(grid)
		vRf = grid - coil[i][0] # Vectors from coil point initial to measurement points
		sRf = np.linalg.norm(vRf, axis = 1) # Magnitudes of vectors

		for k in range(1, len(coil[i])): # For each pairing of points in a particular loop

			vRi = np.copy(vRf)
			sRi = np.copy(sRf)
			vRf = grid - coil[i][k] 
			sRf = np.linalg.norm(vRf, axis = 1)
			sRiRf = np.multiply(sRi, sRf)
			
			vRidotvRf = np.sum(np.multiply(vRi, vRf), axis = 1)
			const = (μ0/(4*np.pi))*currents[i]*(sRi + sRf)/(sRiRf*(sRiRf + vRidotvRf))
			# const = (μ0/4π)*I*(|Ri| + |Rf|)/(|Ri||Rf|(|Ri||Rf| + vRi x vRf))
			temp_field = np.cross(vRi, vRf)*const[:, None]
			coil_field += temp_field

		for j in range(len(grid)):
			field_x[j, i] = coil_field[j, 0]
			field_y[j, i] = coil_field[j, 1]
			field_z[j, i] = coil_field[j, 2]

	
			
			
			
			



	# for k = 2:size(coil{1,sc},2):	# calculate vector of fields in loop over coil segments
	# 	vRi = vRf; sRi=sRf;		# previous final point = new initial point
	# 	vRf = grid - coil{1,sc}(:,k); # calculate new relative vectors
	# 	sRf = sqrt(sum(vRf.*vRf));	# vecnorm(vRf);
	# 	tmp = sRi.*sRf;
	# 	sfield = sfield + cross(vRi,vRf).*(sRi+sRf)./tmp./(tmp + dot(vRi,vRf));		 # Biot-Savart law, see Eq. 9, http://bit.ly/2DsXUTK
	# if size(coil,1)>1 sfield=sfield*coil{2,sc}; end;   #current in second row
	# 	field = field + sfield;
	field = {"Bx": field_x, "By": field_y, "Bz": field_z}
	return field



def main():
	coil1 = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
	coil2 = -1*np.copy(coil1)
	coil3 = 2*np.copy(coil1)

	
	coil = [coil1, coil2, coil3]
	currents = np.array([1, 1, 2])
	coil_list = [coil, currents]



	grid = np.array([1, 1, 1])
	bfield_params = bfield_inputs(coil, grid,  currents=np.array([np.sqrt(2), 2, 3]))
	print(bfield_params)
	field = bfield(bfield_params["coil"], bfield_params["grid"], bfield_params["current"])
	print(field)
	return

if __name__ == "__main__":
	main()
	
