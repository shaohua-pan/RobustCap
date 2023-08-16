"""
    SMPL Renderer. Borrowed from VIBE and ROMP.
"""

__all__ = ['Renderer']

import trimesh
import pyrender
import numpy as np
import pickle


class Renderer:
    def __init__(self, resolution=(224, 224), official_model_file=None):
        """
        Renderer for SMPL model.
        :param resolution: image resolution.
        :param official_model_file: SMPL model file.
        """
        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.resolution = resolution
        self.faces = data['f']
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )
        self.colors = [
            (.7, .7, .6, 1.),
            (.7, .5, .5, 1.),  # Pink
            (.5, .5, .7, 1.),  # Blue
            (.5, .55, .3, 1.),  # capsule
            (.3, .5, .55, 1.),  # Yellow
        ]

    def render(self, image, verts, K, mesh_color=None, mesh_filename=None):
        """
        Render SMPL mesh to the given image.
        :param image: image to render.
        :param verts: SMPL vertices.
        :param K: camera intrinsic matrix.
        :param mesh_color: color of the mesh.
        :param mesh_filename: save mesh to the given file.
        :return:
        """
        triangles = self.faces
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        img_height, img_width = image.shape[:2]
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        camera_pose = np.eye(4)
        camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=camera_pose)
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # for every person in the scene
        mesh = trimesh.Trimesh(verts, triangles)
        mesh.apply_transform(rot)
        if mesh_color is None:
            mesh_color = self.colors[0]
        else:
            mesh_color = mesh_color
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=mesh_color)
        if mesh_filename is not None:
            mesh.export(mesh_filename)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh, 'mesh')
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32)
        valid_mask = (rend_depth > 0)[:, :, None]
        output_image = (color[:, :, :3] * valid_mask +
                        (1 - valid_mask) * image).astype(np.uint8)
        return output_image
