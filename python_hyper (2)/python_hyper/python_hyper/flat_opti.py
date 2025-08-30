import numpy as np
from space_mapper import SpaceMapper
from trimesh import TriMesh
import os
import sys
import argparse
import time
import glob
import numpy as np
# import autograd.numpy as np
# from autograd import grad
from numpy.linalg import svd
import scipy.optimize
from sklearn.decomposition import PCA
from cvxopt import matrix, solvers



import format_loader
from trimesh import TriMesh

from scipy.spatial import procrustes
import plotly.graph_objects as go
prev_Fw_projected = None
def plotting_flat(F_list):
    '''
    PLotting the Falt with Applying Procrustes!
    '''
    global prev_Fw_projected
    F = F_list
    Fw_matrix =F_list
    # print("Constructing Fw_matrix...")
    # for idx, w in enumerate(w_list):
    #     Fw = np.dot(F, w)  # Shape: (12*num_poses,)
    #     if idx < 5:
    #         print(f"  w_list[{idx}] shape: {w.shape}, Fw shape: {Fw.shape}, sample Fw: {Fw[:5]}")
    #     Fw_matrix.append(Fw)

    # Convert to numpy array
    Fw_matrix = np.array(Fw_matrix)  # Shape: (num_vertices, 12*num_poses)
    print("FW: ", Fw_matrix.shape, "FW_matrix dtype:", Fw_matrix.dtype)
    print("Sample of FW_matrix[0,:5]:", Fw_matrix[0,:5])

    # Step 2: Apply PCA to reduce dimensionality to (h-1)
    num_vertices = Fw_matrix.shape[0]
    print(f"Applying PCA to Fw_matrix with n_components={H-1}.")
    Fw_matrix = Fw_matrix.reshape(num_vertices, -1)
    # pca = PCA(n_components=h-1)
    pca = PCA(n_components=H-1)
    Fw_projected = pca.fit_transform(Fw_matrix)  # Shape: (num_vertices, h-1)
    print("FW_proj: ", Fw_projected.shape)
    print("Sample of FW_proj[0,:5]:", Fw_projected[0,:5])

    # Apply Procrustes if we have a previous iteration to compare to
    if prev_Fw_projected is not None and prev_Fw_projected.shape == Fw_projected.shape:
        # Both datasets must have the same shape and ordering
        # Align current iteration's points (Fw_projected) to prev_Fw_projected
        mtx1, mtx2, disparity = procrustes(prev_Fw_projected, Fw_projected)
        # mtx2 is the aligned version of Fw_projected
        Fw_projected = mtx2
        print(f"Applied Procrustes alignment. Disparity: {disparity:.4f}")

    # Update the global variable for the next iteration
    prev_Fw_projected = Fw_projected.copy()

    # Assuming Fw_projected is already defined
    x = Fw_projected[:, 0]
    y = Fw_projected[:, 1]
    z = Fw_projected[:, 2]

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',  # Marker color
            opacity=0.8
        )
    )])

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title='3D Scatter Plot of Points'
    )
    # Show the plot
    fig.show()
def to_spmatrix( M ):
	M = scipy.sparse.coo_matrix( M )
	import cvxopt
	return cvxopt.spmatrix( M.data, np.asarray( M.row, dtype = int ), np.asarray( M.col, dtype = int ) )
def pack( point, B ):
	'''
	`point` is a 12P-by-1 column matrix.
	`B` is a 12P-by-#(handles-1) matrix.
	Returns them packed so that unpack( pack( point, B ) ) == point, B.
	'''
	p12 = B.shape[0]
	handles = B.shape[1]
	X = np.zeros( p12*(handles+1) )
	X[:p12] = point.ravel()
	X[p12:] = B.T.ravel()
	return X

def unpack( X, poses ):
	'''
	X is a flattened array with #handle*12P entries.
	The first 12*P entries are `point` as a 12*P-by-1 matrix.
	The remaining entries are the 12P-by-#(handles-1) matrix B.
	
	where P = poses.
	'''
	P = poses
	point = X[:12*P].reshape(12*P, 1)
	B = X[12*P:].reshape(-1,12*P).T

	return point, B


def normalization_factor_from_xyzs( xyzs ):
	## To make function values comparable, we need to normalize.
	diag = xyzs.max( axis = 0 ) - xyzs.min( axis = 0 )
	diag = np.linalg.norm(diag)
	normalization = 1./( len(xyzs) * diag )
	print( "Normalization of 1/(bounding box diagonal * num-vertices):", normalization )
	return normalization


def optimize_biquadratic( P, H, rest_vs, deformed_vs, x0):
	'''
	Given:
		P: Number of poses
		H: Number of handles
		rest_vs: an array where each row is [ x y z 1 ]
		deformed_vs: an array vertices-by-poses-by-3
		x0: initial guess
	
	If solve_for_rest_pose is False (the default), returns ( converged, final x ).
	If solve_for_rest_pose is True, returns ( converged, final x, and updated rest_vs ).
	'''
	
	## To make function values comparable, we need to normalize.
	normalization = normalization_factor_from_xyzs( rest_vs[:,:3] )
	## We don't want this uniform per-vertex normalization because
	## we may do special weight handling.
	normalization *= len( rest_vs )
	print( "Normalization:", normalization )
	
	
	
	def unpack_W( x, P ):
		pt, B = unpack( x, P )
		## Make sure pt is a column matrix.
		pt = pt.squeeze()[...,np.newaxis]
		assert len( pt.shape ) == 2
		assert pt.shape[1] == 1
		
		## Switch pt with the j-th column of [p;B].
		## 1 Convert [pt;B] from an origin and basis to a set of points.
		W = np.hstack((pt,pt+B))
		
		return W
	
	def pack_W( W ):
		pt = W[:,0:1]
		B = W[:,1:] - pt
		return pack( pt, B )
	
	import flat_metrics
	def W_to_graff( W ):
		## Use a negative threshold so we get back all columns.
		Wgraff = flat_metrics.orthonormalize( np.vstack([ W, np.ones((1,W.shape[1])) ]), threshold = -1 )
		## Keep the same number of columns as the input.
		Wgraff = Wgraff[:,:W.shape[1]]
		return Wgraff
	def W_from_graff( Wgraff ):
		W = Wgraff[:-1] / Wgraff[-1:]
		## This shouldn't be necessary if canonical = False.
		# W = Wgraff[:-1] / np.maximum(Wgraff[-1:], 1e-50) ##### I found a divide by zero error in one running test.

		return W
	
	def W_to_canonical_pB( W ):
		p = W[:,0:1]
		B = W[:,1:] - p
		## First orthonormalize
		B = flat_metrics.orthonormalize( B, threshold = -1 )
		## Then get canonical pt
		p = flat_metrics.canonical_point( p, B )
		return p, B
	def canonical_pB_to_W( p, B ):
		W = np.hstack((p,p+B))
		return W
	def canonical_pB_diff( W0, W1 ):
		p0, B0 = W_to_canonical_pB( W0 )
		p1, B1 = W_to_canonical_pB( W1 )
		pdiff = flat_metrics.distance_between_flats( p0, B0, p1, B1 )
		cosangles = flat_metrics.principal_cosangles( B0, B1, orthonormal = False )
		return pdiff, cosangles
	
	#canonical = 'graff'
	#canonical = 'pB'
	canonical = None
	
	def canonicalize( W ):
		if canonical is None:
			return W
		## Convert W to canonical form by converting in and out of the Graff manifold.
		elif canonical == 'graff':
			return W_from_graff( W_to_graff( W ) )
		elif canonical == 'pB':
			return canonical_pB_to_W( *W_to_canonical_pB( W ) )
		else:
			raise RuntimeError( "Unknown canonical type: %s" % canonical )
	
	## Verify that we can unpack and re-pack shifted without changing anything.
	assert abs( pack_W( unpack_W( np.arange(36), 1 ) ) - np.arange(36) ).max() < 1e-10
	
	
	
	f_eps = 1e-6
	
		## To make xtol approximately match scipy's default gradient tolerance (gtol) for BFGS.
	x_eps = 1e-4
	
	max_iter = 9999
	
	f_zero_threshold = 0.0
	
	
	import flat_intersection_biquadratic_gradients as biquadratic
	
	first_column = None
	
	
	f_prev = None
	
	W_prev = unpack_W( x0.copy(), P )
	W_prev = canonicalize( W_prev )
	
	W = W_prev.copy() ## In case we terminate immediately.
	iterations = 0
	converged = False
	z_f=[]
	try:
		while( True ):
			iterations += 1
			if iterations > max_iter:
				print( "Terminating due to too many iterations: ", max_iter )
				break
			
			print( "Starting iteration", iterations )
			
		
			f = 0
			weights = 0
			
			fis = np.zeros( len( deformed_vs ) )
			W_system = None
			W_rhs = None
			
			
			current_zs=[]
			for i, deformed_v in enumerate(deformed_vs):
				vprime = deformed_v.reshape((3*P,1))
				# print(rest_vs.shape)
				v = rest_vs[i]
			
				v=np.append(v,1)
				## 1
				# print("v: shape: ",v.shape)
				z, ssz, fi = biquadratic.solve_for_z( W_prev, v, vprime, return_energy = True )
				fis[i] = fi
				
				## Save z if that's what we're up to.
				current_zs.append(z)
				ssz = 1.0
				
				## 2
				
				
				## 3
				A, B, Y = biquadratic.linear_matrix_equation_for_W( v, vprime, z)
				if W_system is None:
					W_system, W_rhs = biquadratic.zero_system_for_W( A, B, Y )
				biquadratic.accumulate_system_for_W( W_system, W_rhs, A, B, Y, ssz )
				
				f += fi * ssz
				weights += ssz
			z_f=current_zs
			## 4
			W = biquadratic.solve_for_W_with_system( W_system, W_rhs, first_column = first_column )
			
			W = canonicalize( W )
			
			f *= normalization / weights
			print( "Function value:", f )

			
			print( "Max sub-function value:", fis.max() )
			print( "Min sub-function value:", fis.min() )
			print( "Average sub-function value:", np.average( fis ) )
			print( "Median sub-function value:", np.median( fis ) )
			
			## If this is the first iteration, pretend that the old function value was
			## out of termination range.
			if f_prev is None: f_prev = f + 100*f_eps
			
			if f - f_prev > 0:
				print( "WARNING: Function value increased." )
			if abs( f_prev - f ) < f_eps:
				print( "Function change too small, terminating:", f_prev - f )
				converged = True
				break
		
			
			if f < f_zero_threshold:
				print( "Function below zero threshold, terminating." )
				converged = True
				break
			
			f_prev = f
			W_prev = W.copy()
	
	except KeyboardInterrupt:
		print( "Terminated by KeyboardInterrupt." )
	
	print( "Terminated after", iterations, "iterations." )

#	if csv_path is not None:
#		func_values = np.array( func_values )
#		np.savetxt(csv_path, func_values, delimiter=",")
	# error_recorder.save_error()
	
	print(W.shape,np.array(z_f).shape)
	return converged, W, z_f

OBJ_name = os.path.splitext(os.path.basename("./PerVertex/cube.obj"))[0]
print( "The name for the OBJ is:", OBJ_name )
rest_mesh = TriMesh.FromOBJ_FileName( "./PerVertex/cube.obj" )
rest_vs = np.array( rest_mesh.vs )

print("rest_vs.shape",rest_vs.shape)
rest_vs_original = rest_vs.copy()

pose_paths = glob.glob("./PerVertex/poses-1/cube-*.obj")
pose_paths.sort()
pose_name = os.path.basename( "/PerVertex/poses-1")
print( "The name for pose folder is:", pose_name )
deformed_vs = np.array( [TriMesh.FromOBJ_FileName( path ).vs for path in pose_paths] )
print(deformed_vs.shape)
assert( len(deformed_vs.shape) == 3 )
P, N = deformed_vs.shape[0], deformed_vs.shape[1]
deformed_vs = np.swapaxes(deformed_vs, 0, 1).reshape(N, P, 3)
deformed_vs_original = deformed_vs.copy()
qs_data = np.loadtxt("./PerVertex/qs.txt")
print( "# of good valid vertices: ", qs_data.shape[0] )
H=4

pca = SpaceMapper.Uncorrellated_Space( qs_data, dimension = H )
pt = pca.Xavg_
B = pca.V_[:H-1].T
x0 = pack( pt, B )	
converged, F_list, w_list = optimize_biquadratic( P, H, rest_vs, deformed_vs, x0)
print(converged)
# print(x.shape)
Fw_matrix = []
F = F_list
for w in w_list:
    Fw = np.dot(F, w)  # Shape: (12*num_poses,)
    Fw_matrix.append(Fw)

# Convert to numpy array
Fw_matrix = np.array(Fw_matrix)  # Shape: (num_vertices, 12*num_poses)
print("FW: ",Fw_matrix.shape)
plotting_flat(Fw_matrix)

poses = int(Fw_matrix.shape[1]/12)
for i in range(poses):
	base = "0000"
	name = base[ : -len( str(i+1) ) ] + str(i+1)
	
	per_pose_transformtion = Fw_matrix[:,i*12:(1+i)*12]
	per_pose_transformtion = per_pose_transformtion.reshape(-1,3,4)
	per_pose_transformtion = np.swapaxes(per_pose_transformtion, 1, 2).reshape(-1,12)
	output_path = os.path.join( "./cubeOut", name + ".DMAT" )
	format_loader.write_DMAT( output_path, per_pose_transformtion )

# import os
# import sys
# import argparse
# import time
# import glob
# import numpy as np
# import scipy.optimize
# from numpy.linalg import svd
# from sklearn.decomposition import PCA
# from cvxopt import matrix, solvers
# import scipy.sparse
# import format_loader
# from trimesh import TriMesh
# from scipy.spatial import procrustes
# import plotly.graph_objects as go

# # Ensure that 'space_mapper.py' is in the same directory or in your PYTHONPATH.
# from space_mapper import SpaceMapper

# # Global variable to store the previous PCA-projected points for Procrustes alignment
# prev_Fw_projected = None
# # Global iteration counter for use in the plot title
# iterations = 0

# def plotting_flat(Fw_matrix, H):
#     """
#     Apply PCA to reduce Fw_matrix (per-vertex transformations) to H-1 dimensions,
#     align with the previous iteration using Procrustes (if available), and display a 3D scatter plot.
#     """
#     global prev_Fw_projected, iterations
#     num_vertices = Fw_matrix.shape[0]
#     # Apply PCA to reduce to (H-1) dimensions
#     pca = PCA(n_components=H-1)
#     Fw_projected = pca.fit_transform(Fw_matrix)
    
#     # If previous projection exists, apply Procrustes alignment
#     if prev_Fw_projected is not None and prev_Fw_projected.shape == Fw_projected.shape:
#         mtx1, mtx2, disparity = procrustes(prev_Fw_projected, Fw_projected)
#         Fw_projected = mtx2
#         print(f"Applied Procrustes alignment. Disparity: {disparity:.4f}")
    
#     # Update global variable for next iteration
#     prev_Fw_projected = Fw_projected.copy()
    
#     # If the projected dimension is higher than 3, take only the first 3 dimensions
#     if Fw_projected.shape[1] > 3:
#         Fw_projected = Fw_projected[:, :3]
    
#     # Create the 3D scatter plot
#     fig = go.Figure(data=[go.Scatter3d(
#         x=Fw_projected[:, 0],
#         y=Fw_projected[:, 1],
#         z=Fw_projected[:, 2],
#         mode='markers',
#         marker=dict(
#             size=5,
#             color='blue',
#             opacity=0.8
#         )
#     )])
    
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X Axis',
#             yaxis_title='Y Axis',
#             zaxis_title='Z Axis'
#         ),
#         title=f"Iteration {iterations} Visualization"
#     )
#     # Open the plot in the browser (non-blocking)
#     fig.show(renderer="browser")

# def to_spmatrix(M):
#     M = scipy.sparse.coo_matrix(M)
#     import cvxopt
#     return cvxopt.spmatrix(M.data, np.asarray(M.row, dtype=int), np.asarray(M.col, dtype=int))

# def pack(point, B):
#     """
#     Pack the 12*P column vector 'point' and 12*P-by-(handles-1) matrix B into a flat vector.
#     """
#     p12 = B.shape[0]
#     handles = B.shape[1]
#     X = np.zeros(p12 * (handles + 1))
#     X[:p12] = point.ravel()
#     X[p12:] = B.T.ravel()
#     return X

# def unpack(X, poses):
#     """
#     Unpack X into a column vector 'point' (of shape 12*P-by-1) and a matrix B of shape 12*P-by-(handles-1).
#     """
#     P = poses
#     point = X[:12 * P].reshape(12 * P, 1)
#     B = X[12 * P:].reshape(-1, 12 * P).T
#     return point, B

# def normalization_factor_from_xyzs(xyzs):
#     diag = xyzs.max(axis=0) - xyzs.min(axis=0)
#     diag = np.linalg.norm(diag)
#     normalization = 1. / (len(xyzs) * diag)
#     print("Normalization of 1/(bounding box diagonal * num-vertices):", normalization)
#     return normalization

# def optimize_biquadratic(P, H, rest_vs, deformed_vs, x0, visualize=True):
#     """
#     Optimize the flat using bi-quadratic minimization.
#     Visualize the per-vertex transformations at each iteration if 'visualize' is True.
#     """
#     global iterations
#     normalization = normalization_factor_from_xyzs(rest_vs[:, :3])
#     normalization *= len(rest_vs)
#     print("Normalization:", normalization)
    
#     def unpack_W(x, P):
#         pt, B = unpack(x, P)
#         pt = pt.squeeze()[..., np.newaxis]
#         assert len(pt.shape) == 2 and pt.shape[1] == 1
#         W = np.hstack((pt, pt + B))
#         return W

#     def pack_W(W):
#         pt = W[:, 0:1]
#         B = W[:, 1:] - pt
#         return pack(pt, B)
    
#     # For simplicity, we do not modify canonicalization here.
#     def canonicalize(W):
#         return W

#     f_eps = 1e-6
#     x_eps = 1e-4
#     max_iter = 9999
#     f_zero_threshold = 0.0

#     import flat_intersection_biquadratic_gradients as biquadratic

#     first_column = None
#     f_prev = None
#     W_prev = unpack_W(x0.copy(), P)
#     W_prev = canonicalize(W_prev)
    
#     W = W_prev.copy()
#     iterations = 0
#     converged = False
#     current_zs = []  # Will store per-vertex z values at each iteration
    
#     try:
#         while True:
#             iterations += 1
#             if iterations > max_iter:
#                 print("Terminating due to too many iterations:", max_iter)
#                 break
            
#             print("Starting iteration", iterations)
#             f = 0
#             weights = 0
#             fis = np.zeros(len(deformed_vs))
#             W_system = None
#             W_rhs = None
#             current_zs = []
            
#             for i, deformed_v in enumerate(deformed_vs):
#                 vprime = deformed_v.reshape((3 * P, 1))
#                 v = rest_vs[i]
#                 v = np.append(v, 1)
#                 z, ssz, fi = biquadratic.solve_for_z(W_prev, v, vprime, return_energy=True)
#                 fis[i] = fi
#                 current_zs.append(z)
#                 ssz = 1.0
#                 A, B, Y = biquadratic.linear_matrix_equation_for_W(v, vprime, z)
#                 if W_system is None:
#                     W_system, W_rhs = biquadratic.zero_system_for_W(A, B, Y)
#                 biquadratic.accumulate_system_for_W(W_system, W_rhs, A, B, Y, ssz)
#                 f += fi * ssz
#                 weights += ssz
            
#             f *= normalization / weights
#             print("Function value:", f)
#             print("Max sub-function value:", fis.max())
#             print("Min sub-function value:", fis.min())
#             print("Average sub-function value:", np.average(fis))
#             print("Median sub-function value:", np.median(fis))
            
#             if f_prev is None:
#                 f_prev = f + 100 * f_eps
            
#             if abs(f_prev - f) < f_eps:
#                 print("Function change too small, terminating:", f_prev - f)
#                 converged = True
#                 break
            
#             if f < f_zero_threshold:
#                 print("Function below zero threshold, terminating.")
#                 converged = True
#                 break
            
#             # Visualization: compute per-vertex transformation matrix.
#             # Here, we assume that for each vertex, the transformation is given by:
#             # Fw = W_prev @ z, for each z in current_zs.
#             # Since z is a scalar and W_prev has shape (dim,4), we replicate z into a (4,1) vector.
#             Fw_matrix = np.array([W_prev @ np.array(z).reshape((-1, 1)) for z in current_zs])
#             num_vertices = Fw_matrix.shape[0]
#             Fw_matrix = Fw_matrix.reshape(num_vertices, -1)
#             print("FW_matrix shape for visualization:", Fw_matrix.shape)
#             if visualize:
#                 plotting_flat(Fw_matrix, H)
#                 # Optionally, add a brief pause:
#                 # time.sleep(0.5)
            
#             f_prev = f
#             W_prev = W.copy()
            
#             W = biquadratic.solve_for_W_with_system(W_system, W_rhs, first_column=first_column)
#             W = canonicalize(W)
    
#     except KeyboardInterrupt:
#         print("Terminated by KeyboardInterrupt.")
    
#     print("Terminated after", iterations, "iterations.")
#     print(W.shape, np.array(current_zs).shape)
#     return converged, W, current_zs

# # Main execution
# if __name__ == '__main__':
#     OBJ_name = os.path.splitext(os.path.basename("./PerVertex/cube.obj"))[0]
#     print("The name for the OBJ is:", OBJ_name)
#     rest_mesh = TriMesh.FromOBJ_FileName("./PerVertex/cube.obj")
#     rest_vs = np.array(rest_mesh.vs)
#     print("rest_vs.shape", rest_vs.shape)
#     rest_vs_original = rest_vs.copy()
    
#     pose_paths = glob.glob("./PerVertex/poses-1/cube-*.obj")
#     pose_paths.sort()
#     pose_name = os.path.basename("/PerVertex/poses-1")
#     print("The name for pose folder is:", pose_name)
#     deformed_vs = np.array([TriMesh.FromOBJ_FileName(path).vs for path in pose_paths])
#     print(deformed_vs.shape)
#     assert(len(deformed_vs.shape) == 3)
#     P, N = deformed_vs.shape[0], deformed_vs.shape[1]
#     deformed_vs = np.swapaxes(deformed_vs, 0, 1).reshape(N, P, 3)
#     deformed_vs_original = deformed_vs.copy()
    
#     qs_data = np.loadtxt("./PerVertex/qs.txt")
#     print("# of good valid vertices: ", qs_data.shape[0])
#     H = 4  # Number of handles
    
#     # Initialize PCA-based space mapper from qs_data.
#     pca_mapper = SpaceMapper.Uncorrellated_Space(qs_data, dimension=H)
#     pt = pca_mapper.Xavg_
#     B = pca_mapper.V_[:H-1].T
#     x0 = pack(pt, B)
    
#     converged, F_list, z_list = optimize_biquadratic(P, H, rest_vs, deformed_vs, x0, visualize=True)
#     print("Converged:", converged)
    
#     # After optimization, compute Fw_matrix from the final W (F_list) and the z values.
#     Fw_matrix = np.array([F_list @ np.array(z).reshape((-1, 1)) for z in z_list])
#     Fw_matrix = Fw_matrix.reshape(Fw_matrix.shape[0], -1)
#     print("Final FW_matrix shape:", Fw_matrix.shape)
#     plotting_flat(Fw_matrix, H)
    
#     # Save per-pose transformation files.
#     poses = int(Fw_matrix.shape[1] / 12)
#     for i in range(poses):
#         base = "0000"
#         name = base[:-len(str(i+1))] + str(i+1)
#         per_pose_transformation = Fw_matrix[:, i*12:(i+1)*12]
#         per_pose_transformation = per_pose_transformation.reshape(-1, 3, 4)
#         per_pose_transformation = np.swapaxes(per_pose_transformation, 1, 2).reshape(-1, 12)
#         output_path = os.path.join("./cubeOut", name + ".DMAT")
#         format_loader.write_DMAT(output_path, per_pose_transformation)