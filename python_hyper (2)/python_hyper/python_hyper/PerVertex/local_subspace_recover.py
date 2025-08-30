from __future__ import print_function, division

import numpy as np
import scipy.linalg
import scipy.optimize
from trimesh import TriMesh
# import includes
import time


def v_to_vbar( v, pose_num ):
    '''
    Given:
        v: a 4-vector
        pose_num: an integer
    
    Returns
        The kronecker product of the identity matrix of size 3*pose_num with v as a row matrix:
        np.kron( np.eye( 3*pose_num ), v.reshape(1,-1) )
    
    This is the Vbar and v_expand used in the code.
    '''
    result = np.zeros((3*pose_num,12*pose_num))
    for j in range(pose_num):
        for k in range(3):
            result[j*3+k, (j*3+k)*4:(j*3+k)*4+4]=v
    return result
def solve_directly(V0, V1,use_pseudoinverse = None):
    if use_pseudoinverse is None: use_pseudoinverse = False
    pose_num=V1.shape[1]//3
    left=np.zeros((4,4))
    right=np.zeros((4,3*pose_num))
    constant=0.0
    
    
    for i in range(len(V0)):
        v0=V0[i].reshape(-1,1)
        v1=V1[i].reshape(-1,1)
        left     += v0.dot( v0.T )
        right    += v0.dot( v1.T )
        constant += v1.T.dot(v1).squeeze()
        
        
    
    ssv = np.linalg.norm( left, ord = -2 )
    if ssv < 1e-10: use_pseudoinverse = True
    
   
    
    v0_center = V0[0]
    v1_center = V1[0]
    new_left = np.zeros( ( 5,5 ) )
    new_left[:4,:4] = left
    new_left[:-1,-1] = v0_center.T.squeeze()
    new_left[-1,:-1] = v0_center.squeeze()
    new_right = np.vstack( ( right, v1_center.reshape( (1,-1) ) ) )
    if use_pseudoinverse:
        x_full=np.linalg.pinv(new_left).dot(new_right)
    else:
        x_full=scipy.linalg.solve(new_left,new_right)
  
    x=x_full[:-1].ravel( order='F' )

    y = x.reshape( (4,-1), order='F' )
    cost = (y*left.dot(y)).sum()-2*(right*y).sum()+constant
    return x, cost, ssv

def find_scale(Vertices):
    Dmin=Vertices.min(axis=0)
    Dmax=Vertices.max(axis=0)
    D=Dmax-Dmin
    scale=np.sqrt(D.dot(D))
    return scale


def find_subspace_intersections(rest_pose, other_poses, version=1, method = None, use_pseudoinverse = None, propagate = None, random_sample_close_vertex="euclidean", candidate_num=120, sample_num=10, vertices_num=30, precomputed_geodesic_distance=None):
    
    
    if propagate is None: propagate = False
    
    if method is None:
        method = "vertex"
    print(use_pseudoinverse,propagate, method)
    mesh0=rest_pose
    
    mesh1_list=other_poses
    pose_num=len(mesh1_list)
    
    vertices0=np.hstack((np.asarray(mesh0.vs),np.ones((len(mesh0.vs),1))))
    ## Stack all poses horizontally.
    vertices1 = np.hstack( [ mesh.vs for mesh in mesh1_list ] )
    

    data=vertices0[:,:3]
    vertices_pairwise_distance=np.sqrt(np.square(data.reshape((1,-1,3))-data.reshape((-1,1,3))).sum(axis=-1))
    np.random.seed(1)
    
        
        
    
    
    scale=find_scale(vertices0[:,:3])
    q_space=[]
    errors=[]
    smallest_singular_values=[]

    for i in range(len(vertices1)):
        
    
        distance_to_others = vertices_pairwise_distance[i,:]
        sorted_distance_indices=np.argsort(distance_to_others)
        temp_q_space=[]
        temp_errors=[]
        temp_smallest_singular_values=[]
        for sample in range(sample_num):
            
            random_3_extra_vertices_indices=(np.random.random(vertices_num)*candidate_num).round().astype(np.int32)
            indices=np.asarray(sorted_distance_indices[random_3_extra_vertices_indices])
        
            ## We want everything, we'll use the pseudoinverse.
            # if len(indices)>=3:
            v0=vertices0[i].reshape((1,-1))
            v0_neighbor=vertices0[indices,:]
            v1=vertices1[i].reshape((1,-1))
            v1_neighbor=vertices1[indices,:]
            V0=np.vstack((v0, v0_neighbor))
            V1=np.vstack((v1, v1_neighbor))
            #### solve directly
            q,cost,ssv=solve_directly(V0, V1, use_pseudoinverse = use_pseudoinverse)
            temp_smallest_singular_values.append( ssv )
            assert q is not None
            # if q is not None:
            temp_q_space.append(q)
            temp_errors.append(np.sqrt(np.maximum(cost/(pose_num*scale*scale), 1e-30)))
         
        
        minimum_cost_ind=np.argmin(np.asarray(temp_errors))
        q_space.append(temp_q_space[minimum_cost_ind])
        errors.append(temp_errors[minimum_cost_ind])
        smallest_singular_values.append( temp_smallest_singular_values[minimum_cost_ind] )



    q_space=np.asarray(q_space)
    errors=np.asarray(errors)
    smallest_singular_values=np.asarray(smallest_singular_values)
   
   
    assert len( q_space ) == len( mesh0.vs )
    return q_space, errors, smallest_singular_values



if __name__ == '__main__':
    
    rest_pose_path="cube.obj"
    other_poses_path="./poses-1"
    #### read obj files
    print( "Loading rest pose mesh:", rest_pose_path )
    rest_pose=TriMesh.FromOBJ_FileName(rest_pose_path)
    ## Make sure the mesh is storing arrays.
    rest_pose.vs = np.asarray( rest_pose.vs )
    rest_pose.faces = np.asarray( rest_pose.faces, dtype = int )
    from pathlib import Path

    directory_path = Path(other_poses_path)  # Replace with your directory path
    obj_files = [f.name for f in directory_path.iterdir() if f.is_file() and f.suffix == '.obj']
    obj_files.sort()

    print(obj_files)
    print( "Loading", len( other_poses_path ), "other mesh poses..." )
    other_poses = [ TriMesh.FromOBJ_FileName( other_poses_path+'/'+path ) for path in obj_files ]
    print(len(other_poses))
    for mesh in other_poses:
        mesh.vs = np.asarray( mesh.vs )
        mesh.faces = np.asarray( mesh.faces, dtype = int )
    print( "...done." )

    print( "Generating transformations..." )

    vertices_pairwise_distance=None




    start_time = time.time()
    qs, errors, smallest_singular_values = find_subspace_intersections( rest_pose, other_poses)
    end_time = time.time()
    print( "... Finished generating transformations." )
    print( "Finding subspace intersection duration (minutes):", (end_time-start_time)/60 )
    print("qs shape: ",np.array(qs).shape)
    print (rest_pose.vs.shape)
    print(smallest_singular_values)
    print (find_scale(rest_pose.vs))
    print(qs.shape)
    
    def save_one( path, M ):
        np.savetxt( path, M )
        print( "Saved to path:", path )
    save_one( "./qs.txt", qs )
   