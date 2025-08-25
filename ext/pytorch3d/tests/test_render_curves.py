import unittest
import json

import torch
from pytorch3d import _C
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    PointLights,
)
from pytorch3d.renderer.curve import (
    CurvesRenderer,
    CurveRasterizer,
    CurveRasterizationSettings,
    SilhouetteCompositor,
    rasterize_curves,
    rasterize_curves_python,
)
from pytorch3d.io import load_hair
from pytorch3d.structures import Curves

from matplotlib import pyplot as plt

if __name__ == "__main__":
    from common_testing import TestCaseMixin
else:
    from .common_testing import TestCaseMixin


class TestRasterizeCurves(TestCaseMixin, unittest.TestCase):
    device = torch.device("cuda")
    hair_path = "C:/Users/tangz/Desktop/research/code/pytorch3d/tests/data/test_hair/hair.hair"
    camera_path = "C:/Users/tangz/Desktop/research/code/pytorch3d/tests/data/test_hair/hair_cam.json"

    def _read_curves_and_cameras(self):
        point_list = load_hair(self.hair_path, device=self.device)
        curves = Curves(point_list)
        
        with open(self.camera_path) as f:
            cam_params = json.load(f)
        R, T = look_at_view_transform(
            eye=(cam_params["position"],),
            at=(cam_params["look_at"],),
            up=(cam_params["up"],),
            device=self.device,
        )
        cameras = FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=cam_params["fov"],
            device=self.device,
        )
        
        return curves, cameras
    
    def test_curve_rasterize_visualize(self):
        curves, cameras = self._read_curves_and_cameras()
        
        rasterizer = CurveRasterizer(cameras=cameras)
        curves_proj = rasterizer.transform(curves)
        raster_settings = CurveRasterizationSettings(
                    image_size=64,
                    blur_radius=0.00001,
                    lines_per_pixel=10,
        )

        points_packed = curves_proj.points_packed()
        points_packed.requires_grad_(True)
        
        pix_to_line, zbuf, bary_coords, dists = rasterize_curves(
            curves_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            lines_per_pixel=raster_settings.lines_per_pixel,
            bin_size=raster_settings.bin_size,
            max_lines_per_bin=raster_settings.max_lines_per_bin,
            perspective_correct=True,
            clip_barycentric_coords=True,
        )
        loss = sum([zbuf.sum(), bary_coords.sum(), dists.sum()])
        loss.backward()
        grad = points_packed.grad.clone()
        points_packed.grad.zero_()
        
        # Compare with rasterize_curves_python
        pix_to_line1, zbuf1, bary_coords1, dists1 = rasterize_curves_python(
            curves_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            lines_per_pixel=raster_settings.lines_per_pixel,
            perspective_correct=True,
            clip_barycentric_coords=True,
        )
        loss1 = sum([zbuf1.sum(), bary_coords1.sum(), dists1.sum()])
        loss1.backward()
        grad1 = points_packed.grad.clone()
        assert torch.allclose(grad, grad1, rtol=0.01)
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 3)
        _zbuf = zbuf[..., :3].detach().cpu().numpy().clip(0, 1)
        _zbuf1 = zbuf1[..., :3].detach().cpu().numpy().clip(0, 1)
        axs[0, 0].imshow(_zbuf)
        axs[1, 0].imshow(_zbuf1)
        axs[2, 0].imshow(abs(_zbuf-_zbuf1))
        _bary_coords = bary_coords[..., :3].detach().cpu().numpy().clip(0, 1)
        _bary_coords1 = bary_coords1[..., :3].detach().cpu().numpy().clip(0, 1)
        axs[0, 1].imshow(_bary_coords)
        axs[1, 1].imshow(_bary_coords1)
        axs[2, 1].imshow(abs(_bary_coords-_bary_coords1))
        _dists = dists[..., :3].detach().cpu().numpy().clip(0, 1)
        _dists /= _dists.max()
        _dists1 = dists1[..., :3].detach().cpu().numpy().clip(0, 1)
        _dists1 /= _dists1.max()
        axs[0, 2].imshow(_dists)
        axs[1, 2].imshow(_dists1)
        axs[2, 2].imshow(abs(_dists - _dists1))
        plt.show()
    
    def test_curve_render(self):
        curves, cameras = self._read_curves_and_cameras()
        
        render = CurvesRenderer(
            rasterizer=CurveRasterizer(
                cameras=cameras,
                raster_settings=CurveRasterizationSettings(
                    image_size=4096,
                    blur_radius=0.00005,
                    lines_per_pixel=10,
                    bin_size=0,
                    # max_lines_per_bin=50000,
                )),
            compositor=SilhouetteCompositor()
        )
        
        image = render(curves)
        plt.imshow(image.detach().cpu().numpy())
        plt.show()


if __name__ == "__main__":
    unittest.main()