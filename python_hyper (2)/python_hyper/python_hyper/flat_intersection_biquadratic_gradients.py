"""
Sample code automatically generated on 2018-01-01 04:47:59

by www.matrixcalculus.org

from input

d/dz norm2(V*W*z-vprime)^2 = 2*W'*V'*(V*W*z-vprime)
d/dW norm2(V*W*z-vprime)^2 = 2*V'*(V*W*z-vprime)*z'

where

V is vbar (3p-by-12p)
vprime is a 3p-vector
W is a 12p-by-handles matrix
z is a vector of affine mixing weights for the columns of W (handles)

For the W'*V'*V*W matrix to be invertible, we need
    max(12p, handles, 3p) = max( 3p, handles ) >= handles
which is equivalent to:
    3p >= handles
(Just in case, we can use the pseudoinverse.)

For the formulation finding a V which minimizes the expression,
we de-vectorize W*z into a stack of T 3x3 matrices and t translations and substitute
b=t-vprime:

d/dv norm2(T*v+b)^2 = 2*T'*(b+T*v)

where

T is a 3px3 matrix
b is a 3p-vector
v id a 3p-vector


Another derivation for W and z that is general enough to handle the nullspace version.

E = | A W z - R |^2 = M : M

A = v.T v / |v|^2 (or v)
R = v.T/|v| vprime (or vprime)

dE = 2M : dM
dM = A dW z + A W dz

dE = 2M : ( A dW z + A W dz )
   = 2 * A.T * M * z.T : dW + 2 * (A*W).T * M : dz
   = 2 * A.T * (A W z - R ) * z.T : dW + 2 * (A*W).T * (A W z - R ) : dz
   = 2 * [ A.T*A * W * z * z.T : dW - A.T * R * z.T : dW ]
     + 2 * [ (A*W).T * (A W) z - (A*W).T * R ) : dz

The generated code is provided"as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def unpack( x, poses, handles ):
    W = x.reshape( 12*poses, handles )
    
    assert 12*poses - handles > 0
    
    return W

def pack( W, poses, handles ):
    return W.ravel()

def repeated_block_diag_times_matrix( block, matrix ):
    
    return np.dot( block, matrix.reshape( block.shape[1], -1, order='F' ) ).reshape( -1, matrix.shape[1], order='F' )

def quadratic_for_z( W, v, vprime ):
    '''
    Given:
        W: The 12*P-by-handles array
        v: an array [ x y z 1 ]
        vprime: a 3*P array of all deformed poses
    
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of `z`:
        energy = np.dot( np.dot( z, Q ), z ) + np.dot( L, z ) + C
    '''
    v = v.reshape(1,-1)
    vprime = vprime.squeeze()
    VW = repeated_block_diag_times_matrix( v, W )

    
    Q = np.dot( VW.T, VW )
    L = -2.0*np.dot( vprime, VW )
    C = np.dot( vprime, vprime )
    
    return Q, L, C

def solve_for_z( W, v, vprime,return_energy = False, use_pseudoinverse = True):
    Q, L, C = quadratic_for_z( W, v, vprime)
  
    smallest_singular_value = np.linalg.norm( Q, ord = -2 )
    
    handles = len(L)
    
  
    Qbig = np.zeros( (Q.shape[0]+1, Q.shape[1]+1) )
    Qbig[:-1,:-1] = Q
    Qbig[-1,:-1] = 1
    Qbig[:-1,-1] = 1
    
    rhs = np.zeros( ( len(L) + 1 ) )
    rhs[:-1] = -0.5*L
    rhs[-1] = 1
    z = np.linalg.lstsq( Qbig, rhs, rcond=None )[0][:-1]
   
    
   
    ## This always passes:
    # assert abs( z.sum() - 1.0 ) < 1e-10
    
    if return_energy:
        E = np.dot( np.dot( z, Q ), z ) + np.dot( L, z ) + C
        # print( "New function value after solve_for_z():", E )
        return z, smallest_singular_value, E
    else:
        return z, smallest_singular_value

def quadratic_for_v( W, z, vprime, nullspace = False ):
    '''
    Returns a quadratic expression ( Q, L, C ) for the energy in terms of the 3-vector
    `v`, the rest pose position which can be converted to V bar via:
        kron( identity(poses), kron( identity(3), append( v, [1] ).reshape(1,-1) ) )
        =
        kron( identity(poses*3), append( v, [1] ).reshape(1,-1) )
    
    The quadratic expression returned is:
        energy = np.dot( np.dot( v[:3], Q ), v[:3] ) + np.dot( L, v[:3] ) + C
    '''
    
    if nullspace:
        raise NotImplementedError( "solving for nullspace v is not implemented." )
    
    z = z.squeeze()
    vprime = vprime.squeeze()
    
    assert len( W.shape ) == 2
    assert len( z.shape ) == 1
    assert len( vprime.shape ) == 1
    assert W.shape[1] == z.shape[0]
    
    Taffine = np.dot( W, z ).reshape( -1,4 )
    ## It should be a horizontal stack of poses 3-by-4 matrices.
    assert Taffine.shape[0] % 3 == 0
    ## Separate the left 3x3 from the translation
    T = Taffine[:,:3]
    t = Taffine[:,3]
    
    b = t - vprime
    assert len( b.shape ) == 1
    
    Q = np.dot( T.T, T )
    L = 2.0*np.dot( T.T, b )
    C = np.dot( b, b )
    
    return Q, L, C

def solve_for_v( W, z, vprime, nullspace = False, return_energy = False, use_pseudoinverse = False ):
    Q, L, C = quadratic_for_v( W, z, vprime, nullspace = nullspace )
    
    if use_pseudoinverse:
        # v = np.dot( np.linalg.pinv(Q), -0.5*L )
        v = np.linalg.lstsq( Q, -0.5*L, rcond=None )[0]
    else:
        v = np.linalg.solve( Q, -0.5*L )
    
    ## Restore V to a matrix.
    assert len(vprime) % 3 == 0
    poses = len(vprime)//3
    # V = np.kron( np.identity(poses), np.kron( np.identity(3), np.append( v, [1] ).reshape(1,-1) ) )
    # V = np.kron( np.identity(poses*3), np.append( v, [1] ).reshape(1,-1) )
    result_v = np.append( v, [1] )
    
    if return_energy:
        E = np.dot( np.dot( v, Q ), v ) + np.dot( L, v ) + C
        # print( "New function value after solve_for_V():", E )
        return V, E
    else:
        return V

def linear_matrix_equation_for_W( v, vprime, z ):
    
    
    v = v.squeeze()
    vprime = vprime.squeeze()
    z = z.squeeze()
  
    vprime = vprime.reshape(-1,1)
    z = z.reshape(-1,1)
    
    # A = V'*V = ( I_3poses kron [v 1] )'*( I_3poses kron [v 1] ) = ( I_3poses kron [v 1]' )*( I_3poses kron [v 1] ) = I_3poses kron ( [v 1]'*[v 1] )
    # B' = B = z*z' = z kron z' = z kron z'
    # A kron B' = ( I_3poses kron ( [v 1]'*[v 1] ) ) kron ( z kron z' ) = I_3poses kron( ( [v 1]'*[v 1] ) kron ( z*z' ) )
    ## Reshape v into a row matrix
    v = v.reshape(1,-1)
    
    A = np.dot( v.T, v ) #, np.dot( V.T, V )
    B = np.dot( z, z.T )
    
    Y = np.dot( repeated_block_diag_times_matrix( -v.T, vprime ), z.T )
    
    return A, B, Y

def zero_system_for_W( A, B, Y ):
    system = np.zeros( ( B.shape[1]*A.shape[0], B.shape[0]*A.shape[1] ) )
    rhs = np.zeros( Y.shape )
    
    return system, rhs
    
def accumulate_system_for_W( system, rhs, A, B, Y, weight ):
    system += np.kron( weight*A, B.T )
    rhs -= weight*Y

def solve_for_W( As, Bs, Ys, use_pseudoinverse = True, projection = None, **kwargs ):
    assert len( As ) == len( Bs )
    assert len( As ) == len( Ys )
    assert len( As ) > 0
    
    assert Ys[0].shape[0] % 12 == 0
    poses = Ys[0].shape[0]//12
    system = np.zeros( ( Bs[0].shape[1]*As[0].shape[0], Bs[0].shape[0]*As[0].shape[1] ) )
    ## Our kronecker product formula is the one for row-major vectorization:
    ##     vec( A*X*B ) = kron( A, B.T ) * vec(X)
    ## Since the system matrix is a repeated block diagonal, we can just store
    ## the block and solve against the right-hand-side reshaped with each N entries as
    ## a column.
    ## Vectorize (ravel) the right-hand side at the end.
    rhs = np.zeros( Ys[0].shape )
    for A, B, Y in zip( As, Bs, Ys ):
        ## There is no point to doing this, since the inverse of a block diagonal matrix
        ## is the inverse of each block (and these blocks are repeated).
        # system += np.kron( np.eye( 3*poses ), np.kron( A[0], B.T ) )
        system += np.kron( A, B.T )
        # system += np.kron( A[1], B.T )
        rhs -= Y
    
    return solve_for_W_with_system( system, rhs, use_pseudoinverse = use_pseudoinverse, projection = projection, **kwargs )

def solve_for_W_with_system( system, rhs, use_pseudoinverse = True, projection = None, **kwargs ):
    assert rhs.shape[0] % 12 == 0
    poses = rhs.shape[0]//12
    assert system.shape[0] % 4 == 0
    handles = system.shape[0] // 4
    
    W = np.linalg.lstsq( system, rhs.reshape( -1, system.shape[0] ).T, rcond=None )[0].T.reshape( 12*poses, handles )

    print( "W column norm:", ( ( W - np.average( W, axis = 1 ).reshape(-1,1) )**2 ).sum(axis=0) )

    
    return W

