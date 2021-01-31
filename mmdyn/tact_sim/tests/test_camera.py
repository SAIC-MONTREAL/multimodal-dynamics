import unittest
import numpy as np

from mmdyn.tact_sim.tactile.camera import Camera
from mmdyn.tact_sim.utils.pybullet import setup_pybullet, add_object


TIME_STEP = 1 / 240


class TestCamera(unittest.TestCase):
    def setUp(self):
        add_object(graphic_file="duck.obj", collision_file="duck_vhacd.obj", base_position=[1., 0., 0.2],
                   base_orientation=[0., 0., 0., 1.], mesh_scale=[.1, .1, .1], COM_shift=[0, 0.04, 0.])

        self.cam = Camera(width=640, height=480)
        self.cam.set_view_matrix(camera_eye_pos=[2, 0, 0.5], camera_target_pos=[0, 0, 0.5], camera_up_vec=[0, 0, 1])
        self.cam.set_projection_matrix(fovy=60, aspect=1., near=0.5, far=10)

    def test_camera(self):
        """
        Tests 3D-to-pixel and pixel-to-3D projections.
        """
        x_world_1 = [1, 0, 0]
        x_world_1 = np.reshape(x_world_1, (-1, 1))
        x_cam = self.cam.project_3D_to_pixel(x_world_1)
        x_world_2 = self.cam.unproject_pixel_to_3D(x_cam)

        np.testing.assert_array_almost_equal(x_world_1, x_world_2)

    def test_pointcloud(self):
        """
        Tests pointcloud-to-depthimage and depthimage-to-pointcloud
        and consistency of the `Camera` with PyBullet and OpenGL camera.
        """
        rgb_img_1, depth_img_1, seg_img_1 = self.cam.get_pybullet_image()
        pcl_points, pcl_colors = self.cam.unproject_canvas_to_pointcloud(rgb_img_1, depth_img_1)
        rgb_img_2, depth_img_2 = self.cam.project_pointcloud_to_canvas(pcl_points, pcl_colors)

        # show the depth images
        self.cam.show_image(depth_img_1, RGB=False, title="Image 1")
        self.cam.show_image(depth_img_2, RGB=False, title="Image 2")

        np.testing.assert_array_almost_equal(depth_img_1, depth_img_2)
        np.testing.assert_array_almost_equal(rgb_img_1, rgb_img_2)


if __name__ == '__main__':
    setup_pybullet(time_step=TIME_STEP)

    suite = unittest.TestSuite()
    suite.addTest(TestCamera('test_camera'))
    suite.addTest(TestCamera('test_pointcloud'))
    unittest.TextTestRunner(verbosity=2).run(suite)
