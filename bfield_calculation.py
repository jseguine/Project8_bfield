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
	""" Ensures that the parameters passed to the magnetic field calculating function are in the appropriate form.

	Parameters
	----------
	coil: NDArray, list
		As an NDArray, should have size (m, 3) or (n, m, 3) where n is the number of coils, m is the number of points defining the coils, and 3 are the positions of each coil point.

		If coil is a list, currents must be None. The list should be 2 NDArrays, first describing the coils and second the currents.

	grid: NDArray
		Must have size (3) or (n, 3), where n is the number of gridpoints to compute the field at. 

	currents: NDArray, None
		An NDArray of n currents, corresponding to each of the n coils in order.
		
		If coil is a list, currents must be None, as currents would be defined as part of coil. 

		If coil isn't a list and currents is None, defaults to 1 for each coil.

	Returns
	-------
		dict: 
			- "coil" is an array of discretized coils
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
		grid_dims = np.array([np.copy(grid)])
	else:
		if len(grid[0]) != 3:
			raise ValueError("grid should be 3-vectors")


	# Ensure that coil is usable for this code
	# If coil is a list containing both coil positions and currents, split into separate arrays
	if (type(coil) == list):
		if currents is not None:
			raise TypeError("Can't pass in coil as list and currents")
		if len(coil) != 2:
			raise ValueError("coil as a list should only have an array with all coils and an array of currents.")
		currents = np.copy(coil[1])
		coil = np.copy(coil[0])
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
		"coil": coil_pos,
		"grid": grid,
		"current": currents
	}
	return outputs






def bfield(coil: NDArray, grid: NDArray, currents: NDArray) -> NDArray:
	"""
	Units: coil in m, grid in m, currents in A
	"""
	μ0 = 1.25663706127e-6 # N/A^2
	field = np.zeros_like(grid)			# empty output array of right size
	
	num_coils = len(coil)
	# print(num_coils)
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

def main():
	coil1 = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
	coil2 = -1*np.copy(coil1)
	coil3 = 2*np.copy(coil1)
	# Present limitation, when using multiple coils all must have the same number of points... to fix
	
	coil = np.array([coil1, coil2, coil3])
	currents = np.array([1, 1, 2])
	coil_list = [coil, currents]



	grid = np.array([[1, 1, 1], [1, 0, 1]])
	bfield_params = bfield_inputs(coil, grid, currents=np.array([np.sqrt(2), 2, 3]))
	field = bfield(bfield_params["coil"], bfield_params["grid"], bfield_params["current"])
	print(field)
	return

if __name__ == "__main__":
	main()
	
