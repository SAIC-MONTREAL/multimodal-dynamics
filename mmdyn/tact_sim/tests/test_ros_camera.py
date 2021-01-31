import unittest
from mmdyn.tact_sim.tactile.ros_camera import ROSPinholeCameraModel, ROSStereoCameraModel, ROI


class TestROSCamera(unittest.TestCase):

    def setUp(self):
        self.left_cam = {
            'k': [430.15433020105519, 0.0, 311.71339830549732, 0.0, 430.60920415473657, 221.06824942698509, 0., 0., 1.],
            'd': [-0.363528858080088, 0.16117037733986861, -8.1109585007538829e-05, -0.00044776712298447841, 0.],
            'r': [0.99806560714807102, 0.0068562422224214027, 0.061790256276695904,
                  -0.0067522959054715113, 0.99997541519165112, -0.0018909025066874664,
                  -0.061801701660692349, 0.0014700186639396652, 0.99808736527268516],
            'p': [295.53402059708782, 0.0, 285.55760765075684, 0.0, 0.0, 295.53402059708782,
                  223.29617881774902, 0.0, 0.0, 0.0, 1.0, 0.0],
            'width': 640,
            'height': 480,
        }

        self.right_cam = {
            'k': [428.38163131344191, 0.0, 327.95553847249192, 0.0,
                  428.85728580588329, 217.54828640915309, 0.0, 0.0, 1.0],
            'd': [-0.3560641041112021, 0.15647260261553159, -0.00016442960757099968, -0.00093175810713916221],
            'r': [0.9982082576219119, 0.0067433328293516528, 0.059454199832973849,
                  -0.0068433268864187356, 0.99997549128605434, 0.0014784127772287513,
                  -0.059442773257581252, -0.0018826283666309878, 0.99822993965212292],
            'p': [295.53402059708782, 0.0, 285.55760765075684, -26.507895206214123, 0.0,
                  295.53402059708782, 223.29617881774902, 0.0, 0.0, 0.0, 1.0, 0.0],
            'width': 640,
            'height': 480,
        }

    def test_monocular(self):
        roi = ROI(width=0, height=0, x_offset=0, y_offset=0)
        cam = ROSPinholeCameraModel()
        cam.from_camera_params(**self.left_cam, roi=roi)

        print(cam.rectify_point((0, 0)))
        print(cam.project_3D_to_pixel((0, 0, 10)))

    def test_stereo(self):
        cam = ROSStereoCameraModel()
        cam.from_camera_params(self.left_cam, self.right_cam)

        for x in (16, 320, cam.left_cam.width - 16):
            for y in (16, 240, cam.left_cam.height - 16):
                for d in range(1, 10):
                    pt3d = cam.project_pixel_to_3D((x, y), d)
                    ((lx, ly), (rx, ry)) = cam.project_3D_to_pixel(pt3d)
                    self.assertAlmostEqual(y, ly, 3)
                    self.assertAlmostEqual(y, ry, 3)
                    self.assertAlmostEqual(x, lx, 3)
                    self.assertAlmostEqual(x, rx + d, 3)

        u = 100.0
        v = 200.0
        du = 17.0
        dv = 23.0
        Z = 2.0
        xyz0 = cam.left_cam.project_pixel_to_3DRay((u, v))
        xyz0 = (xyz0[0] * (Z / xyz0[2]), xyz0[1] * (Z / xyz0[2]), Z)
        xyz1 = cam.right_cam.project_pixel_to_3DRay((u + du, v + dv))
        xyz1 = (xyz1[0] * (Z / xyz1[2]), xyz1[1] * (Z / xyz1[2]), Z)
        self.assertAlmostEqual(cam.left_cam.get_delta_u(xyz1[0] - xyz0[0], Z), du, 3)
        self.assertAlmostEqual(cam.left_cam.get_delta_v(xyz1[1] - xyz0[1], Z), dv, 3)
        self.assertAlmostEqual(cam.left_cam.get_delta_x(du, Z), xyz1[0] - xyz0[0], 3)
        self.assertAlmostEqual(cam.left_cam.get_delta_y(dv, Z), xyz1[1] - xyz0[1], 3)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestROSCamera('test_monocular'))
    suite.addTest(TestROSCamera('test_stereo'))
    unittest.TextTestRunner(verbosity=2).run(suite)
