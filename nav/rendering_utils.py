import numpy as np
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
from klampt.math import se3



def render_depth_and_normals(this_vbg,depth,intrinsic,pose,use_depth = False):

    device = o3d.core.Device('CUDA:0')
    intrinsic = intrinsic[:3,:3].astype(np.float64)
    intrinsic = o3c.Tensor(intrinsic.astype(np.float64))
    extrinsic = se3.from_ndarray(pose)
    extrinsic = se3.ndarray(se3.inv(extrinsic))
    extrinsic = o3c.Tensor(extrinsic)
    depth = o3d.t.geometry.Image(depth).to(device)

    if(use_depth):
        block_coords = this_vbg.compute_unique_block_coordinates(
        depth, intrinsic, extrinsic, 1000, 5.0)
    else:
        block_coords = this_vbg.hashmap().key_tensor()

    result = this_vbg.ray_cast(block_coords=block_coords,
                          intrinsic=intrinsic,
                          extrinsic=extrinsic,
                          width=depth.columns,
                          height=depth.rows,
                          render_attributes=[
                              'depth', 'normal', 'index',
                              'interp_ratio'
                          ],
                          depth_scale=1000,
                          depth_min=0,
                          depth_max=5.0,
                          weight_threshold=10,
                          range_map_down_factor=8)
    return result['depth'].cpu().numpy(),result['normal'].cpu().numpy()


    
def get_fov(h,w,fx,fy):
    fovy = 2*np.arctan2(h,(2*fy))
    fovx = 2*np.arctan2(w,(2*fx))
    return fovx,fovy

H,W = (480,640)

def get_camera_rays(H,W,fx,fy):
    fovx,fovy = get_fov(H,W,fx,fy)
    AR = W/H
    u,v = np.meshgrid(np.arange(W),np.arange(H))
    # getting the middle of the pixels
    v = (v+0.5)/H
    u = (u+0.5)/W
    # transforming to centered coordinate frame - and adjusting for the aspect ratio
    u = (2*u-1)*AR
    v = 2*v-1
    # converting to camera coordinates:
    x = u*np.tan(fovx/2)
    y = v*np.tan(fovy/2)
    z = np.ones_like(x)
    rays = np.stack([x,y,z],axis = 2)
    rays = rays/np.linalg.norm(rays,axis =2,keepdims = True)
    rays = rays.reshape(-1,3)
    return rays