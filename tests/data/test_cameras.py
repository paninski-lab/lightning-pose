import torch

from lightning_pose.data.cameras import project_camera_pairs_to_3d


def test_project_camera_pairs_to_3d():

    # ------------------------------------------
    # hard-coded values from anipose-fly example
    # (using aniposelib CameraGroup object)
    # ------------------------------------------

    points = torch.tensor(
        [[[244.00537479, 169.14706428],
         [157.12294343, 177.34581021]],

         [[260.49358356, 119.39195845],
          [168.3701722 , 130.26763126]],

         [[463.32075159, 186.66484865],
          [303.22096047, 260.25092964]]]
    )
    intrinsics = torch.tensor(
        [[[1.4633e+04, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.4633e+04, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.6343e+04, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.6343e+04, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.0100e+05, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.0100e+05, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]]]
    )
    extrinsics = torch.tensor(
        [[[7.9065e-01, -1.3940e-01, 5.9619e-01, -1.4132e+00],
          [-2.8695e-02, 9.6423e-01, 2.6351e-01, -1.0720e+00],
          [-6.1160e-01, -2.2545e-01, 7.5837e-01, 4.7490e+01]],

         [[9.6419e-01, 1.3962e-01, 2.2546e-01, 1.2773e-01],
          [-1.2986e-01, 9.8986e-01, -5.7627e-02, -5.1388e-01],
          [-2.3122e-01, 2.6284e-02, 9.7255e-01, 7.0362e+01]],

         [[7.7300e-01, 4.1197e-01, -4.8244e-01, 3.1492e+00],
          [-3.8902e-01, 9.0852e-01, 1.5250e-01, -1.0950e+00],
          [5.0113e-01, 6.9803e-02, 8.6255e-01, 2.4393e+02]]],
    )
    distortions = torch.tensor(
        [[-21.4957, 0.0000, 0.0000, 0.0000, 0.0000],
         [-14.0726, 0.0000, 0.0000, 0.0000, 0.0000],
         [-1905.7119, 0.0000, 0.0000, 0.0000, 0.0000]],
    )
    target = torch.tensor(
        [[[[-1.59978732, -0.39616025, 3.38618251],
           [-2.05475903, -0.4051827 , 3.45997281]],

          [[-1.59978732, -0.39616025, 3.38618251],
           [-2.05475903, -0.4051827 , 3.45997281]],

          [[-1.59978732, -0.39616025, 3.38618251],
           [-2.05475903, -0.4051827 , 3.45997281]]]]
    )

    p3d = project_camera_pairs_to_3d(
        points=points.unsqueeze(0),
        intrinsics=intrinsics.unsqueeze(0),
        extrinsics=extrinsics.unsqueeze(0),
        dist=distortions.unsqueeze(0),
    )

    assert p3d.shape == (1, 3, 2, 3)  # batch, views, keypoints, coords
    assert torch.allclose(p3d, target, rtol=1e-2)

    # ------------------------------------------
    # test nan-handling
    # ------------------------------------------
    points[0, 0, 0] = float('nan')
    points[0, 0, 1] = float('nan')
    p3d = project_camera_pairs_to_3d(
        points=points.unsqueeze(0),
        intrinsics=intrinsics.unsqueeze(0),
        extrinsics=extrinsics.unsqueeze(0),
        dist=distortions.unsqueeze(0),
    )
    assert p3d.shape == (1, 3, 2, 3)  # batch, view combos, keypoints, coords
    assert torch.all(torch.isnan(p3d[0, 0, 0, :]))
    assert torch.allclose(p3d[0, 0, 1, :], target[0, 0, 1, :], rtol=1e-2)
    assert torch.all(torch.isnan(p3d[0, 1, 0, :]))
    assert torch.allclose(p3d[0, 1, 1, :], target[0, 1, 1, :], rtol=1e-2)
    assert torch.allclose(p3d[0, 2], target[0, 2], rtol=1e-3)
