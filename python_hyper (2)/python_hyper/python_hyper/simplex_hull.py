import os
import sys
import argparse
import time
import numpy
import numpy as np
import scipy

import format_loader
from trimesh import TriMesh
from scipy.spatial import ConvexHull
import glob
from space_mapper import SpaceMapper

########################################
# CMD-line tool for getting filenames. #
########################################
if __name__ == '__main__':
	'''
	Uses ArgumentParser to parse the command line arguments.
	Input:
		parser - a precreated parser (If parser is None, creates a new parser)
	Outputs:
		Returns the arguments as a tuple in the following order:
			(in_mesh, Ts, Tmat)
	'''


	per_vertex_folder = "./cubeOut"
	in_transformations = glob.glob(per_vertex_folder + "/*.DMAT")
	print(len(in_transformations))
	in_transformations.sort()
	Ts = numpy.array([ format_loader.load_DMAT(transform_path).T for transform_path in in_transformations ])	
	num_poses = Ts.shape[0]
	num_verts = Ts.shape[1]
	Ts = numpy.swapaxes(Ts,0,1)
	Ts = Ts.reshape(num_verts, -1)
	
	
	
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	numpy.set_printoptions(precision=4, suppress=True)
	import os,sys
	import mves2
	
	startTime = time.time()
	
	all_Ts = Ts.copy()
	## results stores: volume, solution, iter_num
	
	

	
	Ts_mapper = SpaceMapper.Uncorrellated_Space( Ts, dimension = 3 )
	uncorrelated = Ts_mapper.project( Ts )
	print( "uncorrelated data shape" )
	print( uncorrelated.shape )
	solution, weights, iter_num = mves2.MVES( uncorrelated, method = "qp")
	volume = abs( np.linalg.det( solution ) )
	print(solution.shape)

	print( "=> Best simplex found with volume:", volume )

	
	running_time = (time.time() - startTime)/60
	
	## If we have 4 bones, we can visualize the results in 3D!
	save_json = solution.shape[1] == 4
	if save_json:
		import json
		assert solution.shape[1] == 4
		simplex_vs = solution.T[:,:-1]
		norm_min = simplex_vs.min(axis=0)
		norm_scale = 1./( simplex_vs.max(axis=0) - simplex_vs.min(axis=0) ).max()
		def normalizer( pts ):
			return ( pts - norm_min ) * norm_scale
		
		out_vs = normalizer( uncorrelated )
		simplex_vs = normalizer( simplex_vs )*255.0
		json.dump( { "float_colors": out_vs.tolist() }, open("data.json",'w') )
		json.dump( { "vs": simplex_vs.tolist(), "faces": simplex_vs[ numpy.array([[0,3,2],[3,1,2],[3,0,1],[1,0,2]]) ].tolist() }, open("data-overlay.json",'w') )
	
	recovered = Ts_mapper.unproject( solution[:-1].T )

	
	output_path = os.path.join(per_vertex_folder, "result.txt")
	# if args.output is not None:
	output_path ='./cubeOut/result.txt'
	print( "Saving recovered results to:", output_path )
	format_loader.write_result(output_path, recovered, weights, iter_num, running_time, col_major=True)