import numpy as np


# Transforms
class PointcloudNoise(object):
    """Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    """

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    """Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        
        
        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]

        if "normals" in data.keys():
            normals = data["normals"]
            data_out["normals"] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    """Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        points = data[None]
        occ = data["occ"]

        data_out = data.copy()
        if isinstance(self.N, int):
            if points.ndim == 2:
                idx = np.random.randint(points.shape[0], size=self.N)
                data_out.update(
                    {
                        None: points[idx, :],
                        "occ": occ[idx],
                    }
                )
            elif points.ndim == 3:
                points_list, occ_list = [], []
                for tdx in range(points.shape[0]):
                    idx = np.random.randint(points.shape[1], size=self.N)
                    points_list.append(points[tdx, idx, :][np.newaxis, ...])
                    occ_list.append(occ[tdx, idx][np.newaxis, ...])
                data_out.update(
                    {
                        None: np.concatenate(points_list, axis=0),
                        "occ": np.concatenate(occ_list, axis=0),
                    }
                )
                pass
            else:
                raise RuntimeError("Data loader not support 2020.12.29")
        else:
            Nt_out, Nt_in = self.N
            occ_binary = occ >= 0.5
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update(
                {
                    None: points,
                    "occ": occ,
                    "volume": volume,
                }
            )
        return data_out


class SubsamplePointcloudSeq(object):
    """Point cloud sequence subsampling transformation class.

    It subsamples the point cloud sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    """

    def __init__(self, N, connected_samples=True, random=True):
        self.N = N
        self.connected_samples = connected_samples
        self.random = random

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        data_out = data.copy()
        #points = data[None]  # n_steps x T x 3
        points = data["vertices"]
        n_steps, T, dim = points.shape
        N_max = min(self.N, T)
        if self.connected_samples or not self.random:
            indices = np.random.randint(T, size=self.N) if self.random else np.arange(N_max)
            #data_out[None] = points[:, indices, :]
            data_out["vertices"] = points[:,indices,:]
        else:
            indices = np.random.randint(T, size=(n_steps, self.N))
            data_out[None] = points[np.arange(n_steps).reshape(-1, 1), indices, :]
        return data_out


class SubsamplePointsSeq(object):
    """Points sequence subsampling transformation class.

    It subsamples the points sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    """

    def __init__(self, N, connected_samples=False, random=True):
        self.N = N
        self.connected_samples = connected_samples
        self.random = random

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        points = data[None]
        occ = data["occ"]
        data_out = data.copy()
        n_steps, T, dim = points.shape

        N_max = min(self.N, T)

        if self.connected_samples or not self.random:
            indices = np.random.randint(T, size=self.N) if self.random else np.arange(N_max)
            data_out.update(
                {
                    None: points[:, indices],
                    "occ": occ[:, indices],
                }
            )
        else:
            indices = np.random.randint(T, size=(n_steps, self.N))
            help_arr = np.arange(n_steps).reshape(-1, 1)
            data_out.update({None: points[help_arr, indices, :], "occ": occ[help_arr, indices, :]})
        return data_out

class DownSampleMesh(object):
    """
    Class to downsample mesh using quadric decimation using open3d module

    Args:
    -----
    N : int
        Number of target triangles obtained after downsampling the mesh

    Returns:
    --------
    decimated_mesh : open3d.geometry.TriangleMesh
                    Final downsampled mesh
    """
    def __init__(self,N = 512):
        self.N = N

    def __call__(self, data):
        """
        The method calls the transformation class

        Args:
        -----
        data : Python dictionary
               Contains vertices and triangles as the keys 
        """

        downsampled_mesh = []
        for i in range(data.shape[0]): #TODO : verify the loop
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(data[i]['vertices']) # verify what is the name of the attribute for vertices in data dictionary. It should be vertices as given in load method in MeshField class.
            o3d_mesh.triangles = o3d.utility.Vector3iVector(data[i]['triangles']) # verify what is the name of the attribute for faces in data dictionary. It should be triangles as given in load method in MeshField class.

            decimated_mesh = o3d_mesh.simplify_quadric_Decimation(self.N)
            downsampled_mesh.append(decimated_mesh)

        return downsampled_mesh


class MeshNoise(object):
    """
    The class is used to add noise to the mesh.

    
    Returns:
    --------
    data_out : Python dictionary
               Python dictionary containing vertices and triangles after the noise is added
    mesh_data : open3d.geometry.TriangleMesh
                Mesh after the noise is added
    """

    
    def __call__(self, data):
        """
        The method calls the transformation class

        Args:
        -----
        data : Python dictionary
               Python dictionary containing vertices and triangles before the noise is added
        """
        data_out = data.copy()
        points = data['vertices']

        # TODO : add a loop to add noise to each of the time step model
        noise = self.stddev * np.ones(data_out.shape)
        data_out["vertices"] = points + noise

        mesh_data = o3d.geometry.TriangleMesh()
        mesh_data.vertices = o3d.utility.Vector3dVector(data_out["vertices"])
        mesh_data.triangles = o3d.utility.Vector3iVector(data_out["triangless"])

        return data_out, mesh_data

class PreprocessBase:
    def preprocess(self, shape_x, shape_y):
        raise NotImplementedError()

class PreprocessRotateBase(PreprocessBase):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    
    def create_rotation_matrix(alpha, axis):
        alpha = alpha / 180 * math.pi
        c = torch.cos(alpha)
        s = torch.sin(alpha)
        rot_2d = torch.as_tensor([[c, -s], [s, c]], dtype=torch.float, device=device)
        rot_3d = my_eye(3)
        idx = [i for i in range(3) if i != axis]
        for i in range(len(idx)):
            for j in range(len(idx)):
                rot_3d[idx[i], idx[j]] = rot_2d[i, j]
        return rot_3d

    def _create_rot_matrix(self, alpha):
        return self.create_rotation_matrix(alpha, self.axis)

    def _rand_rot(self):
        alpha = torch.rand(1) * 360
        return self._create_rot_matrix(alpha)

    def rot_sub(self, shape, r):
        if shape.sub is not None:
            for i_p in range(len(shape.sub[0])):
                shape.sub[0][i_p][0, :, :] = torch.mm(shape.sub[0][i_p][0, :, :], r)

        if shape.vert_full is not None:
            shape.vert_full = torch.mm(shape.vert_full, r)

        return shape

    def preprocess(self, shape_x, shape_y):
        raise NotImplementedError()


class PreprocessRotateSame(PreprocessRotateBase):
    def __init__(self, axis=1):
        super().__init__(axis)
        print("Uses preprocessing module 'PreprocessRotateSame'")

    def preprocess(self, shape_x, shape_y):
        r = self._rand_rot()
        shape_x.vert = torch.mm(shape_x.vert, r)
        #shape_y.vert = torch.mm(shape_y.vert, r)

        shape_x = self.rot_sub(shape_x, r)
        #shape_y = self.rot_sub(shape_y, r)
        return shape_x