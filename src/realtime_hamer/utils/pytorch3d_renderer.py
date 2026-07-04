"""PyTorch3D MANO overlay renderer (from ATAboukhadra/hamer-demo)."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from pytorch3d.renderer import (
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    Textures,
)
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.structures.meshes import Meshes

# Left / right colors matching hamer-demo (#858AF1 / #24788F).
_HAND_COLORS = (
    (0.5215686274509804, 0.5411764705882353, 0.9450980392156862),  # left
    (0.1411764705882353, 0.47058823529411764, 0.5607843137254902),  # right
)

_FACES_NEW = np.array(
    [
        [92, 38, 234],
        [234, 38, 239],
        [38, 122, 239],
        [239, 122, 279],
        [122, 118, 279],
        [279, 118, 215],
        [118, 117, 215],
        [215, 117, 214],
        [117, 119, 214],
        [214, 119, 121],
        [119, 120, 121],
        [121, 120, 78],
        [120, 108, 78],
        [78, 108, 79],
    ],
    dtype=np.int64,
)


class MeshPyTorch3DRenderer:
    """Fast full-frame MANO overlay via PyTorch3D."""

    def __init__(
        self,
        faces: np.ndarray,
        device: torch.device,
        render_res: tuple[int, int] | list[int],
        focal_length: float,
    ):
        self.device = device
        self.img_res = (int(render_res[0]), int(render_res[1]))
        self.focal_length = float(focal_length)
        faces = np.concatenate([np.asarray(faces, dtype=np.int64), _FACES_NEW], axis=0)
        self.faces = faces
        self.faces_left = faces[:, [0, 2, 1]].copy()
        self.renderer = self._create_renderer(self.focal_length, self.img_res, device)

    def _create_renderer(self, focal_length: float, render_res: tuple[int, int], device):
        w, h = int(render_res[0]), int(render_res[1])
        cameras = PerspectiveCameras(
            focal_length=((focal_length, focal_length),),
            principal_point=((w / 2.0, h / 2.0),),
            image_size=((h, w),),
            device=device,
            in_ndc=False,
        )
        lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
        raster_settings = RasterizationSettings(
            image_size=(h, w),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
        )

    def maybe_resize(self, render_res: tuple[int, int] | list[int], focal_length: float) -> None:
        res = (int(render_res[0]), int(render_res[1]))
        fl = float(focal_length)
        if res == self.img_res and abs(fl - self.focal_length) < 1e-3:
            return
        self.img_res = res
        self.focal_length = fl
        self.renderer = self._create_renderer(fl, res, self.device)

    def _create_mesh(self, verts: torch.Tensor, faces: torch.Tensor, color) -> Meshes:
        # verts: (1, V, 3), faces: (1, F, 3)
        dummy_verts_uvs = torch.zeros_like(verts[:, :, :2])
        r, g, b = color
        dummy_texture = torch.tensor([[[[r, g, b], [r, g, b]]]], device=verts.device)
        tex = Textures(verts_uvs=dummy_verts_uvs, faces_uvs=faces, maps=dummy_texture)
        return Meshes(verts=verts, faces=faces, textures=tex)

    def vertices_to_trimesh(self, vertices: np.ndarray, camera_translation: np.ndarray, is_right: int):
        faces = self.faces if is_right else self.faces_left
        mesh = trimesh.Trimesh(
            vertices.copy() + camera_translation,
            faces.copy(),
            process=False,
        )
        # hamer-demo: 180° about Z (not X like pyrender path).
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
        mesh.apply_transform(rot)
        return mesh

    @torch.inference_mode()
    def render_rgba(
        self,
        vertices: list[np.ndarray],
        cam_t: list[np.ndarray],
        is_right: list[int] | None = None,
    ) -> np.ndarray:
        """Render all hands; returns float RGBA in [0, 1], shape (H, W, 4)."""
        if is_right is None:
            is_right = [1] * len(vertices)

        meshes = []
        for verts, cam_trans, right in zip(vertices, cam_t, is_right):
            mesh = self.vertices_to_trimesh(verts, cam_trans, is_right=int(right))
            verts_t = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
            faces_t = torch.as_tensor(mesh.faces, dtype=torch.int64, device=self.device)
            color = _HAND_COLORS[int(right)]
            meshes.append(
                self._create_mesh(verts_t.unsqueeze(0), faces_t.unsqueeze(0), color=color)
            )

        scene = join_meshes_as_scene(meshes)
        rendered = self.renderer(scene)[0].detach().float().cpu().numpy()
        return rendered
