# LCA_topology_generator/augmentation.py

import numpy as np
import random

def shear_centerlines(ctrl_points, shear_strength=0.15):
  '''
  Randomly applies shear transformation in the X or Y direction with specified strength

  :param ctrl_points: nx3 array of control points to transform
  :param shear_strength: magnitude of shear in shear matrix
  :return: new_ctrl_points
  '''
  Sx = random.uniform(-shear_strength, shear_strength)
  Sy = random.uniform(-shear_strength, shear_strength)
  Sz = 0
  X_shear = random.getrandbits(1)

  if X_shear:
    shear_matrix = np.array([[1, 0, 0, 0], [Sy, 1, 0, 0], [Sz, 0, 1, 0], [0, 0, 0, 1]])
  else:
    shear_matrix = np.array([[1, Sx, 0, 0], [0, 1, 0, 0], [0, Sz, 1, 0], [0, 0, 0, 1]])

  homogeneous_curve_points = np.concatenate((ctrl_points, np.ones((len(ctrl_points), 1))), axis=1)
  new_ctrl_points = (shear_matrix @ homogeneous_curve_points.T).T[:, 0:-1]

  return new_ctrl_points

def warp1(data_original, ws):
  """
  Randomly warp a curve using a few low frequency sine and cosine modes
    Parameters:
      data_original : nx3 numpy array containing points to be warped
      ws            : warp strength. scalar. How much to warp.

    Returns:
      data          : warped data
  """

  data=data_original.copy()
  
  x,y,z= data[:,0],data[:,1],data[:,2] 

  x-=np.min(x)
  y-=np.min(y)
  z-=np.min(z)

  lx=np.max(x)
  ly=np.max(y)
  lz=np.max(z)

  dx=lx*np.random.uniform(-ws,ws)
  dy=ly*np.random.uniform(-ws,ws)
  dz=lz*np.random.uniform(-ws,ws)

  c1=1.0
  c2=0.2
  c3=0.02
  c4=0.005


  x+=dx*( c1*np.sin((np.pi)*x/lx) + c1*np.cos(np.pi*x/lx) + c2*np.sin(2*np.pi*x/lx) + c3*np.sin(3*np.pi*x/lx)+ c4*np.sin(5*np.pi*x/lx))
  y+=dy*( c1*np.sin((np.pi)*y/ly) + c1*np.cos(np.pi*y/ly) + c2*np.sin(2*np.pi*y/ly) + c3*np.sin(3*np.pi*y/ly)+ c4*np.sin(5*np.pi*y/ly))
  z+=dz*( c1*np.sin((np.pi)*z/lz) + c1*np.cos(np.pi*z/lz) + c2*np.sin(2*np.pi*z/lz) + c3*np.sin(3*np.pi*z/lz)+ c4*np.sin(5*np.pi*z/lz))

  return data

def apply_tree_shear(tree_ctrl_points: np.ndarray, strength: float = 0.12, rng: np.random.Generator = None) -> np.ndarray:
    """
    Applies coordinate shearing to the entire (27, 3) LCA tree.
    Because shearing applies a uniform linear transformation to all points,
    snapped control points (bifurcation at indices 4, 5, and 17) will remain perfectly snapped.
    
    :param tree_ctrl_points: Control point tree array of shape (27, 3).
    :param strength: Magnitude of shear in the shear matrix. Default is 0.12 (matching reference).
    :param rng: A numpy random generator instance (unused here as reference relies on random package).
    :return: Sheared control points of shape (27, 3).
    """
    # Simply call the reference implementation
    return shear_centerlines(tree_ctrl_points, shear_strength=strength)

def apply_tree_warp(tree_ctrl_points: np.ndarray, strength: float = 0.10, rng: np.random.Generator = None) -> np.ndarray:
    """
    Applies wave warping to the entire (27, 3) LCA tree using low frequency sine/cosine modes.
    Since the warp displacement depends only on the coordinate values, identical input points
    (like indices 4, 5, and 17) will receive the identical shift, preserving the snapped bifurcation.
    
    :param tree_ctrl_points: Control point tree array of shape (27, 3).
    :param strength: Magnitude of warping. Default is 0.10 (matching reference).
    :param rng: A numpy random generator instance (unused here as reference relies on np.random).
    :return: Warped control points of shape (27, 3).
    """
    # Simply call the reference implementation
    return warp1(tree_ctrl_points, ws=strength)
