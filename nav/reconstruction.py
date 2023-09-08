"""
Heavily inspired by https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/integrate_custom.py
"""

import numpy as np
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
import cv2
from klampt.math import se3
import pickle
import torch
import pdb
from rendering_utils import render_depth_and_normals,get_camera_rays
import torch.utils.dlpack
from torch import linalg as LA
import torch.nn as nn
import time
init_blocks = 50000

def get_intrinsics(W,H):
    focal_length = W/(2*np.tan((79/2)*np.pi/180))
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
#     intrinsics.set_intrinsics(width = W, height = H, fx = focal_length,fy = focal_length,cx = W//2,cy = H//2)
#     intrinsics_matrix = intrinsics.intrinsic_matrix
    intrinsics_matrix = np.zeros((3,3))
    intrinsics_matrix[[0,1],[0,1]] = focal_length
    intrinsics_matrix[2,2] = 1
    intrinsics_matrix[0,2] = W/2
    intrinsics_matrix[1,2] = H/2
    return intrinsics_matrix

def get_properties(voxel_grid,points,attributes,res = 8,voxel_size = 0.025,device = o3d.core.Device('CUDA:0')):
    """ This function returns the coordinates of the voxels containing the query points specified by 'points' and their respective attributes
    stored the 'attribute' attribute within the voxel_block_grid 'voxel_grid'

    Args:
        voxel_grid (open3d.t.geometry.VoxelBlockGrid): The voxel block grid containing the attributes and coordinates you wish to extract
        points (np.array [Nx3]- dtype np.float32): array containing the XYZ coordinates of the points for which you wish to extract the attributes in global coordinates
        attribute list [(str)]: the string corresponding to the attribute you wish to obtain within your voxel_grid (say, semantic label, color, etc)
        res (int, optional): Resolution of the dense voxel blocks in your voxel block grid.  Defaults to 8.
        voxel_size (float, optional): side length of the voxels in the voxel_block_grid  Defaults to 0.025.
    """
    if(points.shape[0]>0):
        # we first find the coordinate of the origin of the voxel block of each query point and turn it into an open3d tensor
        start = time.time()
        query = np.floor((points/(res*voxel_size))).astype(np.int32)
        t_query = o3c.Tensor(query.astype(np.int32),device = device)

        # we then find the remainder between the voxel block origin and each point to find the within-block coordinates of the voxel containing this point
        query_remainder = points-query*res*voxel_size
        query_remainder_idx = np.floor(query_remainder/voxel_size).astype(np.int32)
        qri = query_remainder_idx

        # we then obtain the hashmap 
        hm = voxel_grid.hashmap()

        # we extract the unique voxel block origins for memory reasons and save the mapping for each individual entry
        # pdb.set_trace()
        # pickle.dump(query,open('debug_query.p','wb'))
        # block_c,mapping = np.unique(query,axis = 0,return_inverse = True)
        block_c,mapping = torch.unique(torch.from_numpy(query).to('cuda:0'),dim = 0,return_inverse = True)
        block_c = block_c.cpu().numpy()
        mapping = mapping.cpu().numpy()

        t_block_c = o3c.Tensor(block_c.astype(np.int32),device = device)
        start = time.time()

        # we query the hashmap to find the index corresponding to each of the voxel block origins
        r,m = hm.find(t_block_c)

        # we then find the flattened indices and coordinates of the individual voxels within each of these voxel blocks in memory
        coords,indices = voxel_grid.voxel_coordinates_and_flattened_indices(r.to(o3c.int32))
    #     print(mapping)


        # we then reshape the index array for easier querying according to the voxel block resolution
        idx = indices.reshape((-1,res,res,res)).cpu().numpy()
        
        # finally, we find the corresponding memory flattened index of the voxels containing the query points, remembering that the slicing order
        # for the dense voxel block is z-y-x for some reason. 
        # we make sure to clip the values, for robustness sake - no ecc is a pain in the ...
        qri[:,2] = np.clip(qri[:,2],0,idx.shape[1]-1)
        qri[:,1] = np.clip(qri[:,1],0,idx.shape[2]-1)
        qri[:,0] = np.clip(qri[:,0],0,idx.shape[3]-1)

        selected_idx = idx[mapping,qri[:,2],qri[:,1],qri[:,0]]
        # we do the same for the coordinates
        coords = coords.reshape((-1,res,res,res,3))
        selected_coords = coords[mapping,qri[:,2],qri[:,1],qri[:,0],:]
        all_attrs = {}
        for attribute in attributes:
        # we then extract the attribute we wish to query from the voxel block grid and flatten it
            attr = voxel_grid.attribute(attribute)
            attr = attr.reshape((-1,attr.shape[-1]))
            all_attrs.update({attribute:attr[selected_idx,:]})


        #finally, we return the selected attributes for those indices, as weel as the coordinates of the voxels containing the query points
        return all_attrs,selected_coords
    else: 
        return None,points

def get_indices_from_points(voxel_grid,points,res = 8,voxel_size = 0.025,device = o3d.core.Device('CUDA:0')):
    """ This function returns the indices of the points designated by points

    Args:
        voxel_grid (open3d.t.geometry.VoxelBlockGrid): The voxel block grid containing the attributes and coordinates you wish to extract
        points (np.array [Nx3]- dtype np.float32): array containing the XYZ coordinates of the points for which you wish to extract the attributes in global coordinates
        attribute (str): the string corresponding to the attribute you wish to obtain within your voxel_grid (say, semantic label, color, etc)
        res (int, optional): Resolution of the dense voxel blocks in your voxel block grid.  Defaults to 8.
        voxel_size (float, optional): side length of the voxels in the voxel_block_grid  Defaults to 0.025.
    """
    if(points.shape[0]>0):
        # we first find the coordinate of the origin of the voxel block of each query point and turn it into an open3d tensor
        query = np.floor((points/(res*voxel_size)))
        t_query = o3c.Tensor(query.astype(np.int32),device = device)

        # we then find the remainder between the voxel block origin and each point to find the within-block coordinates of the voxel containing this point
        query_remainder = points-query*res*voxel_size
        query_remainder_idx = np.floor(query_remainder/voxel_size).astype(np.int32)
        qri = query_remainder_idx

        # we then obtain the hashmap 
        hm = voxel_grid.hashmap()

        # we extract the unique voxel block origins for memory reasons and save the mapping for each individual entry
        block_c,mapping = np.unique(query,axis = 0,return_inverse = True)
        t_block_c = o3c.Tensor(block_c.astype(np.int32),device = device)

        # we query the hashmap to find the index corresponding to each of the voxel block origins
        r,m = hm.find(t_block_c)

        # we then find the flattened indices and coordinates of the individual voxels within each of these voxel blocks in memory
        coords,indices = voxel_grid.voxel_coordinates_and_flattened_indices(r.to(o3c.int32))

        # we then reshape the index array for easier querying according to the voxel block resolution
        idx = indices.reshape((-1,res,res,res)).cpu().numpy()
        
        # finally, we find the corresponding memory flattened index of the voxels containing the query points, remembering that the slicing order
        # for the dense voxel block is z-y-x for some reason. 
        selected_idx = idx[mapping,qri[:,2],qri[:,1],qri[:,0]]
    return selected_idx

def get_COLORS():
    COLORS = np.array([
    [0,0,0],[151,226,173],[174,198,232],[31,120,180],[255,188,120],[188,189,35],[140,86,74],[255,152,151],[213,39,40],[196,176,213],[148,103,188],[196,156,148],
    [23,190,208],[247,183,210],[218,219,141],[254,127,14],[227,119,194],[158,218,229],[43,160,45],[112,128,144],[82,83,163]
    ]).astype(np.uint8)
    return COLORS


class Reconstruction:

    def __init__(self,depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,trunc_multiplier = 8,n_labels = 10,integrate_color = True,device = o3d.core.Device('CUDA:0'),miu = 0.001):
        """Initializes the TSDF reconstruction pipeline using voxel block grids, ideally using a GPU device for efficiency. 

        Args:
            depth_scale (float, optional): Describes the conversion factor of your depth image to meters - defaults to 1000:1 (i.e. each unit of depth is 1/1000 m). Defaults to 1000.0.
            depth_max (float, optional): Maximum depth reading in meters. Defaults to 5.0 m.
            res (int, optional): The number of voxels per locally connected block in the voxel block grid . Defaults to 8.
            voxel_size (float, optional): The size of the voxels in the voxel grid, in meters. Defaults to 0.025.
            n_labels (_type_, optional): Number of semantic labels in the semantic map. Leave as None if not doing metric-semantic reconstruction. When provided, performs to metric semantic reconstruction. Defaults to None.
            integrate_color (bool, optional): Whether or not to add color to the reconstructed mesh. If false, color informaton is not integrated. Defaults to True.
            device (_type_, optional): Which (CPU or GPU) you wish to use to performs the calculation. CUDA devices ~strongly~ encouraged for performance. Defaults to o3d.core.Device('CUDA:0').
            miu (float, optional): Laplace smoothing factor used to ensure numeric stability in metric-semantic reconstruction. Defaults to 0.001.
        """
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.res = res
        self.voxel_size = voxel_size
        self.n_labels = n_labels
        self.integrate_color = integrate_color
        self.device = device
        self.semantic_integration = self.n_labels is not None
        self.miu = miu
        self.trunc = self.voxel_size * trunc_multiplier
        self.trunc_multiplier = trunc_multiplier
        self.initialize_vbg()
        self.rays = None
        self.torch_device = torch.device('cuda')
        self.weight_sizes = None

    def initialize_vbg(self):
        if(self.integrate_color and (self.n_labels is None)):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'),
            (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)),
            self.voxel_size,self.res, init_blocks, self.device)
        elif((self.integrate_color == False) and (self.n_labels is not None)):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'label'),
            (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (self.n_labels)),
            self.voxel_size,self.res, init_blocks, self.device)
        elif((self.integrate_color) and (self.n_labels is not None)):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight','color','label'),
            (o3c.float32, o3c.float32, o3c.float32,o3c.float32), ((1), (1),(3),(self.n_labels)),
            self.voxel_size,self.res, init_blocks, self.device)
        else:
            print('No color or Semantics')
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight'),
            (o3c.float32, o3c.float32), ((1), (1)),
            self.voxel_size,self.res, init_blocks, self.device)

    def verify_contents_not_empty(self,weight_threshold = 3.0):
        # def print_active_entries(hashmap):
        active_buf_indices = self.vbg.hashmap().active_buf_indices().to(o3c.int32)
        if(len(active_buf_indices)>0):
            voxel_coords, voxel_indices = self.vbg.voxel_coordinates_and_flattened_indices(active_buf_indices)
            weight = self.vbg.attribute('weight').reshape((-1, 1))
            valid_weights = weight[voxel_indices] >= weight_threshold
            return np.any(valid_weights.cpu().numpy())

        else:
            return False


    def update_vbg(self,depth,intrinsic,pose,color = None,semantic_label = None):
        """Adds a new observation to the metric (or metric-semantic) map

        Args:
            depth (ndarray - HxWx1): Depth image as a numpy array
            intrinsic (ndarray - 3x3): Intrinsic matrix of the depth camera (supposes the color image has the same intrinsics)
            pose (ndarray - 4x4 np.float64): The camera's transform w.r.t. the world frame
            color (ndarray - np.uint8 HxWx3, optional): The color image of the observation. Must be present if performing colored metric reconstruction. Defaults to None.
            semantic_label (ndarray - HxWxn_labels np.float32, optional): The current observed logits for the semantic segmentation of this map. Must be present if performing metric-semantic reconstruction_description_. Defaults to None.
        """
        self.depth = depth
        self.pose = pose

        if(self.rays is None):
            # print('calculating rays')
            self.rays =  torch.from_numpy(get_camera_rays(depth.shape[0],depth.shape[1],intrinsic[0,0],intrinsic[1,1])).to(self.torch_device)
        self.intrinsic = intrinsic
        intrinsic = o3c.Tensor(intrinsic.astype(np.float64))
        depth = o3d.t.geometry.Image(depth).to(self.device)
        extrinsic = se3.from_ndarray(pose)
        extrinsic = se3.ndarray(se3.inv(extrinsic))
        extrinsic = o3c.Tensor(extrinsic)#extrinsics[i]
        # Get active frustum block coordinates from input
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, self.depth_scale,
            self.depth_max)
        
        # Activate them in the underlying hash map (may have been inserted)
        self.vbg.hashmap().activate(frustum_block_coords)
        self.active_frustum_block_coords = frustum_block_coords

        # Find buf indices in the underlying engine
        buf_indices, masks = self.vbg.hashmap().find(frustum_block_coords)
        o3d.core.cuda.synchronize()
        voxel_coords, voxel_indices = self.vbg.voxel_coordinates_and_flattened_indices(
            buf_indices)
        o3d.core.cuda.synchronize()

        # Now project them to the depth and find association
        # (3, N) -> (2, N)
        extrinsic_dev = extrinsic.to(self.device, o3c.float32)
        self.current_extrinsic = extrinsic_dev
        self.intrinsic = intrinsic
        xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3,
                                                                    3:]
        intrinsic_dev = intrinsic.to(self.device, o3c.float32)
        uvd = intrinsic_dev @ xyz
        d = uvd[2]
        u = (uvd[0] / d).round().to(o3c.int64)
        v = (uvd[1] / d).round().to(o3c.int64)
        o3d.core.cuda.synchronize()
        mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (
            v < depth.rows)

        v_proj = v[mask_proj]
        u_proj = u[mask_proj]
        d_proj = d[mask_proj]
        depth_readings = depth.as_tensor()[v_proj, u_proj, 0].to(
            o3c.float32) / self.depth_scale
        sdf = depth_readings - d_proj

        mask_inlier = (depth_readings > 0) \
            & (depth_readings < self.depth_max) \
            & (sdf >= -self.trunc)

        sdf[sdf >= self.trunc] = self.trunc
        sdf = sdf / self.trunc
        o3d.core.cuda.synchronize()

        weight = self.vbg.attribute('weight').reshape((-1, 1))
        if(self.weight_sizes is None):
            self.weight_sizes = weight.shape
        # print(weight.shape,self.weight_sizes)
        if(self.weight_sizes[0] != weight.shape[0]):
            print('oops, weight changed shape from {} to {}'.format(self.weight_sizes,weight.shape))
            self.weight_sizes = weight.shape
        tsdf = self.vbg.attribute('tsdf').reshape((-1, 1))

        valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
        w = weight[valid_voxel_indices]
        wp = w + 1

        tsdf[valid_voxel_indices] \
            = (tsdf[valid_voxel_indices] * w +
            sdf[mask_inlier].reshape(w.shape)) / (wp)
        
        o3d.core.cuda.synchronize()

        if(self.integrate_color):

            self.update_color(color,depth,valid_voxel_indices,mask_inlier,w,wp,v_proj,u_proj)
        if(self.semantic_integration):
            self.update_semantics(semantic_label,v_proj,u_proj,valid_voxel_indices,mask_inlier,weight)
        weight[valid_voxel_indices] = wp
        o3d.core.cuda.synchronize()
        

    def update_color(self,color,depth,valid_voxel_indices,mask_inlier,w,wp,v_proj,u_proj):
        #performing color integration
        color = cv2.resize(color,(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
        color = o3d.t.geometry.Image(o3c.Tensor(color.astype(np.float32))).to(self.device)
        color_readings = color.as_tensor()[v_proj,u_proj].to(o3c.float32)
        color = self.vbg.attribute('color').reshape((-1, 3))
        color[valid_voxel_indices] \
                = (color[valid_voxel_indices] * w +
                            color_readings[mask_inlier]) / (wp)
        o3d.core.cuda.synchronize()
        
    def update_semantics(self,semantic_label,v_proj,u_proj,valid_voxel_indices,mask_inlier,weight):
        # performing semantic integration
        #  Laplace Smoothing of the observation
        semantic_label += self.miu
        renormalizer = 1+self.miu*self.n_labels
        semantic_label = semantic_label/renormalizer

        semantic_label = np.log(semantic_label)

        semantic_image = o3d.t.geometry.Image(semantic_label).to(self.device)
        
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        # initializing previously unobserved voxels with uniform prior
        semantic[valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]] += o3c.Tensor(np.log(np.array([1.0/self.n_labels])).astype(np.float32)).to(self.device)
        #Bayesian update in log space    
        semantic[valid_voxel_indices] = semantic[valid_voxel_indices]+semantic_readings[mask_inlier]
        o3d.core.cuda.synchronize()

    def extract_visibile_metric_point_cloud(self,weight_threshold = 1.0):
        start = time.time()
        cpu_device =  o3d.core.Device('CPU:0')
        extrinsics = self.current_extrinsic.to(o3c.float64).to(cpu_device)
        W = 640
        H = 480
        intrinsics = get_intrinsics(W,H)
        intrinsics = o3c.Tensor(intrinsics.astype(np.float64))

        active_block_coords = self.active_frustum_block_coords
        # print('before rendering took {}'.format(time.time()-start))
        start = time.time()
        res = self.vbg.ray_cast(active_block_coords,intrinsic = intrinsics,extrinsic = extrinsics,width = W,height =H,
                   depth_min = 0.5,depth_max = 5,render_attributes = ['depth'],range_map_down_factor = 8,depth_scale =1,
                   weight_threshold = weight_threshold,trunc_voxel_multiplier = self.trunc_multiplier)
        # print('rendering took {}'.format(time.time()-start))
        start = time.time()
        depth = o3d.t.geometry.Image(res.depth)
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth,intrinsics,extrinsics,depth_scale = 1,depth_max = 5.0)
        # print('point cloud creation took {}'.format(time.time()-start))
        return pcd
        

    def extract_point_cloud(self,return_raw_logits = False,weight_threshold = 1.0,visible = True):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        if(visible):
            pcd = self.extract_visibile_metric_point_cloud(weight_threshold = weight_threshold)
        else:
            pcd = self.vbg.extract_point_cloud(weight_threshold = weight_threshold)
        pcd = pcd.to_legacy()
        sm = nn.Softmax(dim = 1)
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            # labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            # weights = get_properties(self.vbg,target_points,'weight',res = self.res,voxel_size = self.voxel_size,device = self.device)[0]
            property_dict,coords = get_properties(self.vbg,target_points,['label','weight'],res = self.res,voxel_size = self.voxel_size,device = self.device)

            if property_dict is not None:
                labels = property_dict['label']
                weights = property_dict['weight']
                labels = labels.cpu().numpy().astype(np.float64)
                weights = weights.cpu().numpy()
                if(return_raw_logits):
                    return pcd,labels,weights
                else:
                    labels = labels
                    labels = sm(torch.from_numpy(labels)).numpy()
                    return pcd,labels,weights
            else:
                return None,None,None
        else:
            return pcd,None,weights

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        sm = nn.Softmax(dim =1)
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            # labels = self.precision_check(labels)
            vertex_labels = sm(torch.from_numpy(labels)).numpy()
            #getting the correct probabilities
            return mesh,vertex_labels
        else:
            return mesh,None
    def precision_check(self,labels):
        #compensating for machine precision of exponentials by setting the maximum log to 0 
        labels += -labels.max(axis = 1).reshape(-1,1)
        return labels
    def save_vbg(self,path):
        self.vbg.save(path)


class NaiveAveragingReconstruction(Reconstruction):

    def initialize_vbg(self):
        if(not self.integrate_color):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight','label','semantic_weight'),
            (o3c.float32, o3c.float32, o3c.float32,o3c.float32), ((1), (1), (self.n_labels),(1)),
            self.voxel_size,self.res, init_blocks, self.device)
            self.original_size = self.vbg.attribute('label').shape[0]
            # print('\n\nstarting without color\n\n')
        else:
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight','label','semantic_weight','color'),
            (o3c.float32, o3c.float32, o3c.float32,o3c.float32,o3c.float32), ((1), (1), (self.n_labels),(1),(3)),
            self.voxel_size,self.res, init_blocks, self.device)
            self.original_size = self.vbg.attribute('label').shape[0]
            # print('\n\n\n starting with color \n\n\n')

    def update_semantics(self, semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight):


        semantic_label = semantic_label

        semantic_image = o3d.t.geometry.Image(semantic_label).to(self.device)
        
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        semantic_weight = self.vbg.attribute('semantic_weight').reshape((-1))
        # initializing previously unobserved voxels with uniform prior
        #naive summing of probabilities
        semantic[valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]] += o3c.Tensor(np.array([1.0/self.n_labels]).astype(np.float32)).to(self.device)
        semantic_weight[valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]] += o3c.Tensor(0.2).to(o3c.float32).to(self.device)

        #Bayesian update in log space    
        semantic[valid_voxel_indices] = (semantic_weight[valid_voxel_indices].reshape((-1,1))*semantic[valid_voxel_indices]+semantic_readings[mask_inlier])/(semantic_weight[valid_voxel_indices].reshape((-1,1))+1)
        semantic_weight[valid_voxel_indices] += 1
        o3d.core.cuda.synchronize()



    def extract_point_cloud(self,return_raw_logits = False,weight_threshold = 1.0,visible = True):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        if(visible):
            start = time.time()
            pcd = self.extract_visibile_metric_point_cloud(weight_threshold = weight_threshold)
            # print('Extracting metric point cloud took {}'.format(time.time()-start))
        else:
            pcd = self.vbg.extract_point_cloud(weight_threshold = weight_threshold)        # pdb.set_trace()
        pcd = pcd.to_legacy()
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            # labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            # # print('getting labels took {}'.format(time.time()-start))
            # weights = get_properties(self.vbg,target_points,'weight',res = self.res,voxel_size = self.voxel_size,device = self.device)[0]
            property_dict,coords = get_properties(self.vbg,target_points,['label','weight'],res = self.res,voxel_size = self.voxel_size,device = self.device)

            if property_dict is not None:
                labels = property_dict['label']
                weights = property_dict['weight']
                weights = weights.cpu().numpy()
                if(return_raw_logits):
                    return pcd,labels.cpu().numpy().astype(np.float64)
                else:
                    labels = labels.cpu().numpy().astype(np.float64)
                    return pcd,labels,weights
            else:
                return None,None,None
        else:
            return pcd,None,weights

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            vertex_labels = labels
            return mesh,vertex_labels
        else:
            return mesh,None

class GroundTruthGenerator(Reconstruction):
    def __init__(self,depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,trunc_multiplier = 8,n_labels = None,integrate_color = True,device = o3d.core.Device('CUDA:0'),miu = 0.001):
        super().__init__(depth_scale,depth_max,res,voxel_size,trunc_multiplier,n_labels,integrate_color,device,miu)

    def update_semantics(self,semantic_label,v_proj,u_proj,valid_voxel_indices,mask_inlier,weight):
        "takes in the GT mask resized to the depth image size"
        # now performing semantic integration
        # semantic_label = cv2.resize(data_dict['semantic_label'],(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
        # semantic_label = model.classify(data_dict['color'],data_dict['depth'])
        # cv2.resize(semantic_label,(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
        # print(np.max(semantic_label),np.min(semantic_label))
        # one-hot encoding semantic label
        semantic_label = torch.nn.functional.one_hot(torch.from_numpy(semantic_label.astype(np.int64)),num_classes = self.n_labels).numpy().astype(np.float32)
        #  Laplace Smoothing #1
        color = o3d.t.geometry.Image(semantic_label).to(self.device)
        
        color_readings = color.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        color = self.vbg.attribute('label').reshape((-1, self.n_labels))
        # Detection Count update
        color[valid_voxel_indices]  = color[valid_voxel_indices]+color_readings[mask_inlier]
    def extract_triangle_mesh(self,return_raw_logits = False):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,['label'],res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels['label']
            labels = labels.cpu().numpy().astype(np.float64)
            #getting the correct probabilities
            if(return_raw_logits):
                vertex_labels = labels
            else:
                vertex_labels = labels/labels.sum(axis=1,keepdims = True)
            return mesh,vertex_labels
        else:
            return mesh,None
    def extract_point_cloud(self,return_raw_logits = False,weight_threshold = 1.0,visible = True):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """


        if(visible):
            pcd = self.extract_visibile_metric_point_cloud(weight_threshold = weight_threshold)
        else:
            pcd = self.vbg.extract_point_cloud(weight_threshold = weight_threshold)
        pcd = pcd.to_legacy()
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            weights = get_properties(self.vbg,target_points,'weight',res = self.res,voxel_size = self.voxel_size,device = self.device)[0]
            if labels is not None:
                weights = weights.cpu().numpy()
                labels = labels.cpu().numpy().astype(np.float64)
                labels[labels.sum(axis =1)==0] = 1/21.0

                # pdb.set_trace()
                if(return_raw_logits):
                    labels = labels
                else:
                    labels = labels/labels.sum(axis=1,keepdims = True)
                return pcd,labels
            else:
                return None,None
        else:
            return pcd,None

class HistogramReconstruction(GroundTruthGenerator):
    def update_semantics(self, semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight):
        semantic_label = np.argmax(semantic_label,axis = 2)
        # semantic_label = torch.nn.functional.one_hot(torch.from_numpy(semantic_label.astype(np.int64)),num_classes = self.n_labels).numpy().astype(np.float32)
    #  Laplace Smoothing #1
        semantic_image =  o3d.t.geometry.Image(semantic_label).to(self.device)
                
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                                u_proj].to(o3c.int64)
        # pdb.set_trace()
        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        # initializing previously unobserved voxels with uniform prior

        # updating the histogram
        semantic[valid_voxel_indices,semantic_readings[mask_inlier].flatten()] = semantic[valid_voxel_indices,semantic_readings[mask_inlier].flatten()]+1
        
        o3d.core.cuda.synchronize()

class GeometricBayes(Reconstruction):
    def extract_point_cloud(self,return_raw_logits = False,weight_threshold = 1.0,visible = True):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """

        if(visible):
            pcd = self.extract_visibile_metric_point_cloud(weight_threshold = weight_threshold)
        else:
            pcd = self.vbg.extract_point_cloud(weight_threshold = weight_threshold)
        pcd = pcd.to_legacy()
        target_points = np.asarray(pcd.points)
        sm = nn.Softmax(dim = 1)
        if(self.semantic_integration):
            property_dict,coords = get_properties(self.vbg,target_points,['label','weight'],res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = property_dict['label']
            weights = property_dict['weight']
            # labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            # weights,coords= get_properties(self.vbg,target_points,'weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            weights = weights.cpu().numpy().astype(np.float64)
            if labels is not None:
                if(return_raw_logits):
                    return pcd,(labels/weights)
                else:
                    labels = (labels/weights)
                    labels[np.isnan(labels)] = 1
                    labels = sm(torch.from_numpy(labels)).numpy()
                    #getting the correct probabilities
                    return pcd,labels,weights
            else:
                return None,None,None
        else:
            return pcd,None,weights

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        sm = nn.Softmax(dim=1)
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            weights,coords= get_properties(self.vbg,target_points,'weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = (labels/weights).cpu().numpy()
            labels[np.isnan(labels)] = 1
            labels = sm(torch.from_numpy(labels)).numpy()
            vertex_labels = labels
            return mesh,vertex_labels
        else:
            return mesh,None

class GeneralizedIntegration(Reconstruction):
    def __init__(self, depth_scale=1000, depth_max=5, res=8, voxel_size=0.025, trunc_multiplier=8, n_labels=None, integrate_color=True, device=o3d.core.Device('CUDA:0'), 
                 miu=0.001,epsilon = 0,L = 0,torch_device = 'cuda:0',T=np.array(1)):
        super().__init__(depth_scale, depth_max, res, voxel_size, trunc_multiplier, n_labels, integrate_color, device, miu)
        self.epsilon = epsilon
        self.L = L
        self.torch_device = torch_device
        self.T = torch.from_numpy(T).to(self.torch_device)
        self.sm = nn.Softmax(dim = 2)
    def initialize_vbg(self):
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
        ('tsdf', 'weight', 'log_label','label','semantic_weight'),
        (o3c.float32, o3c.float32, o3c.float32,o3c.float32,o3c.float32), ((1), (1), (self.n_labels),(self.n_labels),(1)),
        self.voxel_size,self.res, init_blocks, self.device)
        self.original_size = self.vbg.attribute('label').shape[0]
    def get_epsilon_and_L(self,semantic_label=0,semantic_readings=0):
        epsilon = o3c.Tensor([self.epsilon]).to(self.device)
        L = o3c.Tensor([self.L]).to(self.device)
        return epsilon.reshape((1,1)).to(o3c.float32),L.reshape((1,1)).cpu().numpy()
    def get_weights(self,semantic_label,semantic_readings,v_proj,u_proj,valid_voxel_indices,mask_inlier):
        w = o3c.Tensor(np.ones(shape = valid_voxel_indices.shape)).to(o3c.float32).to(self.device)
        # print(w.shape)
        return w.reshape((-1,1))
    def get_temperatures(self,semantic_label):
        return self.T.view(1,1,-1)
    def update_semantics(self, semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight):
        new_size = self.vbg.attribute('label').shape[0]
        if(new_size != self.original_size):
            print('VBG size changed from {} to {}'.format(self.original_size,new_size))
            self.original_size = new_size

        # Scaling Step
        T = self.get_temperatures(semantic_label)

        semantic_label = torch.from_numpy(semantic_label).to(self.torch_device)
        semantic_label = self.sm(semantic_label/T)
        # Laplace smoothing step
        semantic_label += self.miu
        renormalizer = 1+self.miu*self.n_labels
        semantic_label = semantic_label/renormalizer
        semantic_label_l = torch.log(semantic_label)
        semantic_label_l = semantic_label_l.cpu().numpy()
        semantic_label = semantic_label.cpu().numpy()

        # projection step
        semantic_image = o3d.t.geometry.Image(semantic_label).to(self.device)
        semantic_image_l = o3d.t.geometry.Image(semantic_label_l).to(self.device)
        
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        semantic_readings_l = semantic_image_l.as_tensor()[v_proj,u_proj].to(o3c.float32)

        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        semantic_l = self.vbg.attribute('log_label').reshape((-1, self.n_labels))
        semantic_weights = self.vbg.attribute('semantic_weight').reshape((-1))

        W = self.get_weights(semantic_label,semantic_readings,v_proj,u_proj,valid_voxel_indices,mask_inlier)
        # epsilon,L = self.get_epsilon_and_L(semantic_label,semantic_readings)
        o3d.core.cuda.synchronize()

        # initializing previously unobserved voxels with uniform prior
        # for log operations
        new_voxels = valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]
        semantic_l[new_voxels] += o3c.Tensor(np.log(np.array([1.0/self.n_labels])).astype(np.float32)).to(self.device)
        # for averaging
        semantic[new_voxels] += o3c.Tensor(np.array([1.0/self.n_labels]).astype(np.float32)).to(self.device)
        # initializing weights to one:
        semantic_weights[new_voxels] = o3c.Tensor(np.array([1.0])).to(o3c.float32).to(self.device)
        o3d.core.cuda.synchronize()

        #Bayesian update in log space    
        semantic_l[valid_voxel_indices] = semantic_l[valid_voxel_indices]+W*semantic_readings_l[mask_inlier]
        o3d.core.cuda.synchronize()

        #Bayesian update in non-log space
        # print(semantic_readings[mask_inlier].cpu().numpy().isnull())
        semantic[valid_voxel_indices] = semantic[valid_voxel_indices]+W*semantic_readings[mask_inlier]
        o3d.core.cuda.synchronize()

        #Updating the Semantic Weights:
        # print(semantic_weights[valid_voxel_indices])
        semantic_weights[valid_voxel_indices] = semantic_weights[valid_voxel_indices] +  W.reshape((-1))
        o3d.core.cuda.synchronize()
        o3d.core.cuda.release_cache()
        # return super().update_semantics(semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight)

    def extract_point_cloud(self,return_raw_logits = False,weight_threshold = 1.0,visible = True):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        if(visible):
            pcd = self.extract_visibile_metric_point_cloud(weight_threshold = weight_threshold)
        else:
            pcd = self.vbg.extract_point_cloud(weight_threshold = weight_threshold)
        pcd = pcd.to_legacy()
        sm = nn.Softmax(dim = 1)
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels_l,coords = get_properties(self.vbg,target_points,'log_label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            semantic_weights,coords = get_properties(self.vbg,target_points,'semantic_weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            epsilon,L = self.get_epsilon_and_L()
            alpha = (1-epsilon)/(semantic_weights.reshape((-1,1))) + epsilon
            alpha = alpha.cpu().numpy()
            labels = labels.cpu().numpy().astype(np.float64)
            labels_l = labels_l.cpu().numpy().astype(np.float64)   
            if labels is not None:
                if(return_raw_logits):
                    return pcd,labels,labels_l,semantic_weights.cpu().numpy()
                else:
                    labels = alpha*labels
                    labels_l = alpha*labels
                    l_probs = sm(torch.from_numpy(labels_l)).numpy()
                    probs = labels/labels.sum(axis = 1,keepdims = True)
                    final_probs = L*l_probs+(1-L)*probs
                    final_probs = final_probs/final_probs.sum(axis =1,keepdims = True)

                    return pcd,final_probs
            else:
                return None,None
        else:
            return pcd,None

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        sm = nn.Softmax(dim =1)
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            log_labels,coords = get_properties(self.vbg,target_points,'log_label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            semantic_weights,coords = get_properties(self.vbg,target_points,'semantic_weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            epsilon,L = self.get_epsilon_and_L()
            alpha = (1-epsilon)/(semantic_weights) + epsilon
            alpha = alpha.cpu().numpy()
            labels = labels.cpu().numpy().astype(np.float64)
            labels_l = labels_l.cpu().numpy().astype(np.float64)   

            if labels is not None:
                labels = alpha*labels
                labels_l = alpha*labels
                l_probs = sm(torch.from_numpy(labels_l)).numpy()
                probs = labels/labels.sum(axis = 1,keepdims = True)
                final_probs = L*l_probs+(1-L)*probs
                final_probs = final_probs/final_probs.sum(axis =1,keepdims = True)

                return mesh,final_probs
        else:
            return mesh,None

class LearnedGeneralizedIntegration(GeneralizedIntegration):
    def __init__(self, depth_scale=1000, depth_max=5, res=8, voxel_size=0.025, trunc_multiplier=8, n_labels=None, integrate_color=True, device=o3d.core.Device('CUDA:0'),
                  miu=0.001, epsilon=0, L=0, torch_device='cuda:0', T=np.array(1),weights = np.array(1),depth_ranges = np.arange(0.0,5.1,0.5),angle_ranges =  np.arange(0,90.1,30)):
        super().__init__(depth_scale, depth_max, res, voxel_size, trunc_multiplier, n_labels, integrate_color, device, miu, epsilon, L, torch_device, T)
        self.weights = weights
        self.weights[self.weights <0] = 0
        self.depth_ranges = depth_ranges
        self.angle_ranges = angle_ranges

    def compute_and_digitize(self,rendered_depth_1,n1):
        with torch.no_grad():
            device = self.torch_device
            this_rays = self.rays
            dr = torch.from_numpy(self.depth_ranges).to(device)
            ar = torch.from_numpy(self.angle_ranges).to(device)
            n = torch.from_numpy(n1).to(device)
            # pdb.set_trace()

            rendered_depth = torch.from_numpy(rendered_depth_1).to(device)
            n = torch.clamp(n,-1,1)
            n[torch.all(n == 0,dim = 2)] = torch.Tensor([0,0,1.0]).to(device)
            digitized_depth = torch.clamp(torch.bucketize(rendered_depth[:,:,0].float()/1000,dr),0,dr.shape[0]-2)
            p = (n.view(-1,3)*this_rays).sum(axis = 1)
            p = torch.clamp(p/(torch.linalg.norm(n.view(-1,3),dim =1)*torch.linalg.norm(this_rays,dim =1)),-1,1)
            projective_angle = torch.arccos((torch.abs(p)))*180/np.pi
            # print('this is ar shape = {} and ar = {}'.format(ar.shape[0]-1,ar))
            angle_proj = torch.clamp(torch.bucketize(projective_angle,ar).reshape(digitized_depth.shape),0,ar.shape[0]-2)
            # pdb.set_trace()

            del dr 
            del ar 
            del n
            del p
            del rendered_depth

            return digitized_depth.cpu().numpy(),angle_proj.cpu().numpy() 

    def get_weights(self,semantic_label,semantic_readings,v_proj_o3d,u_proj_o3d,valid_voxel_indices,mask_inlier_03d):
        sl = semantic_label.argmax(axis=2)
        v_proj = v_proj_o3d.cpu().numpy()
        u_proj = u_proj_o3d.cpu().numpy()
        mask_inlier = mask_inlier_03d.cpu().numpy()
        rendered_depth,n = render_depth_and_normals(self.vbg,self.depth,self.intrinsic,self.pose,use_depth = True)
        digitized_depth,digitized_angle = self.compute_and_digitize(rendered_depth,n)
        # pdb.set_trace()
        selected_weights = self.weights[sl[v_proj,u_proj],digitized_depth[v_proj,u_proj],digitized_angle[v_proj,u_proj]]
        return o3c.Tensor(selected_weights).to(self.device).reshape((-1,1)).to(o3c.float32)[mask_inlier]

class PeanutMapper():
    def __init__(self,args,n_classes = 10,voxel_size = 0.025,device =  o3d.core.Device('CUDA:0'),intrinsic = None,cuda_device = None,depth_scale = 1.0):
        self.rec_type = args.fusion_type
        self.device = device
        self.voxel_size = voxel_size
        self.n_classes = n_classes
        self.intrinsic = intrinsic
        self.args = args
        self.verified = False
        self.weight_threshold = 2
        self.res = 8
        self.integrate_color = True
        self.starting_pose = None
        self.trunc_multiplier = 8
        if(cuda_device is None):
            self.cuda_device = args.device = torch.device("cuda:" + str(args.sem_gpu_id) if args.cuda else "cpu")
        else:
            self.cuda_device = cuda_device
        assert self.rec_type in ['Averaging','Bayesian','Histogram','Geometric'],"""Chosen 
        Reconstruction type {} is not in the allowed reconstruction type list 
        of ['Averaging','Bayesian','Histogram','Geometric']""".format(self.rec_type)
        if(self.rec_type == 'Averaging'):
            self.rec_class = NaiveAveragingReconstruction
        elif(self.rec_type == 'Histogram'):
            self.rec_class = HistogramReconstruction
        elif(self.rec_type == 'Bayesian'):
            self.rec_class = Reconstruction
        elif(self.rec_type == 'Geometric'):
            self.rec_class = GeometricBayes
        else:
            raise NotImplementedError('Yo Bro, this is not implemented, is there a typo in the rec type? How did this get through the assertion?')
        
        self.rec = self.rec_class(res = self.res,voxel_size = self.voxel_size,n_labels = self.n_classes,device = self.device,depth_scale = depth_scale,integrate_color = self.integrate_color,trunc_multiplier = self.trunc_multiplier)
        if(intrinsic is None):
            focal_length = 640/(2*np.tan((79/2)*np.pi/180))
            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(width = 640, height = 480, fx = focal_length,fy = focal_length,cx = 320,cy = 240)
            self.intrinsics = intrinsics.intrinsic_matrix

    def update_vgb(self,outer_obs,info):
        obs = outer_obs.squeeze()
        rgb = obs[:3,:,:].permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
        depth = (obs[3:4,:,:]/100).permute(1,2,0).contiguous().cpu().numpy()
        # print(np.unique(depth,return_counts = True))
        semseg = obs[4:,:,:].permute(1,2,0).contiguous().cpu().numpy()
        # print(depth.max(),depth.min(),info)
        # pdb.set_trace()
        if(self.starting_pose is None):
            self.starting_pose = info
        info = info-self.starting_pose
        gps = info[:2]
        angle = info[2]
        pose = self.get_pose_from_gps_compass(gps,angle)
        # if()
        #filtering dogshit frames
        if((depth>self.args.max_depth).sum()/depth.flatten().shape < 0.98):
            self.rec.update_vbg(depth,self.intrinsics,pose,rgb,semseg)
        del outer_obs
        torch.cuda.empty_cache()

        # pass
    def update_and_get_map(self,obs,info,old_map):
        self.update_vgb(obs,info)
        new_map = self.get_map(old_map)
        new_map[2:4,:,:] = torch.clone(old_map[2:4,:,:])
        # new_map[1,:,:] = torch.maximum(new_map[1,:,:],old_map[1,:,:])
        #making sure traced paths are always available
        passed_by_there = new_map[2:4,:,:].any(dim = 0)
        new_map[1,:,:][passed_by_there] = 1
        new_map[0,:,:][passed_by_there] = 0
        new_map[1,:,:] = torch.maximum(old_map[1,:,:],new_map[1,:,:])

        # new_map[1,:,:][new_map[2:4,:,:].any(du)]
        return new_map
    
    def get_color_map_debug(self,ground_labels,digitized_X,digitized_Y,obstacle):
        objectness = (ground_labels[:,:,4:13] > 0).any(axis = 2)
        vacant = (ground_labels.sum(axis =2) == 0)
        mlo = ground_labels[:,:,4:13].argmax(axis = 2)
        mlc = ground_labels[:,:,4:14].argmax(axis = 2)
        mlc[digitized_Y[obstacle],digitized_X[obstacle]] = 15
        mlc[objectness] = mlo[objectness]
        mlc[vacant] = 21
        mlc = mlc.cpu().detach().numpy()
        COLORS = get_COLORS()
        COLORS[0] = np.array([196,176,213])
        COLORS[9] = np.array([231,231,231])
        COLORS[15] = np.array([157,157,157])
        COLORS = np.concatenate((COLORS,np.array([255,255,255]).reshape(1,-1)))
        color_map = COLORS[mlc].astype(np.uint8)
        # print(color_map.shape,color_map.dtype)
        cv2.imshow('My_map',color_map)
        cv2.waitKey(1)
        
        
        # color_map = cv2.flip(color_map,0)
        pass

    def get_map(self,old_map):
        with torch.no_grad():
            if(not self.verified):
                self.verified = self.rec.verify_contents_not_empty(weight_threshold = self.weight_threshold)
            if(self.verified):
                start = time.time()
                pcd,labels,weights = self.rec.extract_point_cloud(weight_threshold = self.weight_threshold,visible = False)
                # print('extracting point cloud took {}'.format(time.time()-start))
            else:
                labels = None
            xrange = torch.from_numpy(np.arange(-self.args.map_size_cm/200,self.args.map_size_cm/200,self.args.map_resolution/100)).to(self.cuda_device)
            if(labels is not None):
                # self.rec.vbg.save('debug_vbg.npz')
                # pickle.dump(self.rec.active_frustum_block_coords,open('frustum_block_coords.p','wb'))
                # pickle.dump(self.rec.intrinsic,open('instrinsics.p','wb'))
                # pickle.dump(self.rec.current_extrinsic,open('current_extrinsics.p','wb'))
                # o3d.visualization.draw_geometries([pcd])
                # o3d.io.write_point_cloud('./debug_pcd.pcd', pcd, write_ascii=False, compressed=True, print_progress=False)
                labels = torch.from_numpy(labels).to(self.cuda_device)
                weights = torch.from_numpy(weights).to(self.cuda_device).flatten()
                # o3d.visualization.draw_geometries([pcd])
                pcd_t = torch.from_numpy(np.asarray(pcd.points)).to(self.cuda_device)
                # pdb.set_trace()
                height_mask = pcd_t[:,1] >-5
                pcd_t = pcd_t[height_mask]
                weights = weights[height_mask]
                pcd_t[:,0] = -pcd_t[:,0]
                pcd_t[:,2] = pcd_t[:,2]
                thold = 0.2
                thold_pred = self.args.map_trad_detection_threshold
                uncertain_thold = 0.3
                obstacle_weight_threshold = 1

                labels = labels[height_mask]
                thold_labels = (labels > thold).any(axis = 1)
                hard_labels =labels.argmax(dim =1) + 4
                top_labels = labels.max(dim = 1).values.float()
                hard_labels[~thold_labels] = labels.shape[1]
                # hard_labels = torch.Tensor(hard_labels).long().to(self.cuda_device)
                hard_labels = hard_labels.long()
                # digitized_pcd = torch.bucketize(pcd_t,xrange,right = False)
                digitized_pcd = torch.round(torch.clamp(xrange.shape[0]*(pcd_t/(self.args.map_size_cm/100) + 0.5),0,xrange.shape[0]-1)).long()

                Z = -pcd_t[:,1]

                del pcd_t
                torch.cuda.empty_cache()

                digitized_Y = digitized_pcd[:,0]
                digitized_X = digitized_pcd[:,2]
                obstacle_high = Z<self.args.camera_height + 0.3
                obstacle_low = Z>0.3
                # downward_stairs = Z < -0.3
                obstacle_obs = weights>=obstacle_weight_threshold
                obstacle = torch.logical_and(obstacle_high,obstacle_low)
                obstacle = torch.logical_and(obstacle,obstacle_obs)
                # obstacle = torch.logical_or(obstacle,downward_stairs)
                # pdb.set_trace()
                # digitized_X = torch.Tensor(np.digitize(X,xrange)).long()
                # digitized_Y = torch.Tensor(np.digitize(Y,xrange)).long()
                # color_map = torch.zeros(size = (xrange.shape[0],xrange.shape[0],3))
                # class_map = torch.zeros(size = (xrange.shape[0],xrange.shape[0]))
                ground_confidences = torch.zeros(size = (xrange.shape[0],xrange.shape[0],labels.shape[1]+4),device = self.cuda_device)
                ground_counts = torch.zeros(size = (xrange.shape[0],xrange.shape[0],labels.shape[1]+4),device = self.cuda_device)
                ground_counts.index_put_((digitized_Y,digitized_X,hard_labels),torch.ones(labels.shape[0],device=self.cuda_device),accumulate = True)
                ground_confidences.index_put_((digitized_Y,digitized_X,hard_labels),top_labels,accumulate = True)
                ground_labels = ground_confidences/ground_counts
                ground_labels = torch.nan_to_num(ground_labels,nan = 0,posinf = 0,neginf = 0)

                valid_detections = (ground_labels[:,:,4:13] > thold_pred)
                objectness = valid_detections.any(axis =2)
                # ground_labels[:,:,13][objectness] = 0
                # ground_labels[:,:,4:13][valid_detections] = 1
                # ground_labels[:,:,4:13][torch.logical_not(objectness)] = 0
                #clipping for compatibility with
                # ground_labels[:,:,4:13][objectness] = 1
                #ensuring potential objects arent overwritten by background
                # ground_labels[:,:,13][objectness] = 0

                # vacant = (ground_labels.sum(axis =2) == 0)
                explored = torch.logical_and(ground_counts.sum(axis = 2)>0,torch.any(ground_labels[:,:,4:]>thold_pred,dim = 2))
                uncertain = torch.logical_and(ground_labels[:,:,4:13] > uncertain_thold,ground_labels[:,:,4:13] < thold_pred).any(axis = 2)
                ground_labels[digitized_Y[obstacle],digitized_X[obstacle],0] = 1     
                ground_labels[:,:,0][uncertain] = 0
                # ground_labels.index_put_((digitized_Y[obstacle],digitized_X[obstacle],torch.zeros(digitized_X[obstacle].shape[0],device=self.cuda_device).long()),torch.ones(digitized_X[obstacle].shape[0],device=self.cuda_device),accumulate = True)
                # ground_labels[:,:,0] = ground_labels[:,:,0] > 5
                # ground_labels[objectness][:,0] = 0
                ground_labels[:,:,1][explored] = 1
                ground_labels[:,:,1][uncertain] = 0     

                # ground_labels[:,:,1][uncertain] = 0
                # ground_labels[:,:,0][uncertain] = 0
                # ground_labels[:,:,4:] = ground_labels[:,:,4:] >thold_pred
                # self.get_color_map_debug(ground_labels,digitized_X,digitized_Y,obstacle)
                ground_labels = ground_labels.permute(2,0,1)
                del Z
                del digitized_pcd
                del obstacle_high
                del obstacle_low
                del obstacle
                del objectness
                del digitized_X
                del digitized_Y
                del ground_confidences
                del ground_counts
                del weights
                del height_mask
                del obstacle_obs
                # del uncertain
                # del vacant
                torch.cuda.empty_cache()
                o3d.core.cuda.release_cache()

            else:
                ground_labels = torch.zeros(size = (xrange.shape[0],xrange.shape[0],self.args.num_sem_categories+4),device = self.cuda_device)
                ground_labels = ground_labels.permute(2,0,1)
            
            return ground_labels

    def get_rotation_matrix_from_compass(self,theta):
        R = np.zeros((3,3))
        R[1,1] = 1
        R[0,0] = np.cos(-theta)
        R[0,2] = np.sin(-theta)
        R[2,0] = -np.sin(-theta)
        R[2,2] = np.cos(-theta)
        return R
    
    def get_pose_from_gps_compass(self,gps,compass):
        pose = np.eye(4)
        pose[:3,:3] = self.get_rotation_matrix_from_compass(compass)
        pose[0,3] = gps[1]
        pose[1,3] = -0.88
        pose[2,3] = gps[0]
        return pose


def main():
    voxel_size = 0.025
    trunc = voxel_size * 8
    res = 8
    depth_scale = 1000.0
    depth_max = 5.0
    root_dir = "/home/motion/data/scannet_v2"
    color = False
    data = pickle.load(open('example_data_reconstruction.p','rb'))

    ### Performing metric reconstruction only
    rec = Reconstruction(depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,n_labels = None,integrate_color = False,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    for data_dict in tqdm(data):
        depth = data_dict['depth']
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
        # pcd,_ = rec.extract_point_cloud()
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])



    ### Performing Colored Metric reconstruction
    rec = Reconstruction(depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,n_labels = None,integrate_color = True,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    total_len = len(data)

    for data_dict in tqdm(data):
        depth = data_dict['depth']
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])




    del rec
    ### performing colored metric-semantic reconstruction
    rec = Reconstruction(depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,n_labels = 21,integrate_color = True,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    for data_dict in tqdm(data):
        depth = data_dict['depth']
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])
    print(label,label.shape)



if __name__ == '__main__':
    main()


