import torch

from lightning_pose.data.cameras import (
    get_valid_projection_masks,
    project_camera_pairs_to_3d,
)


def test_project_camera_pairs_to_3d():

    # hard-coded values from anipose-fly example (using aniposelib CameraGroup object)

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


def test_get_valid_projection_masks():

    n_batch = 2
    n_views = 3
    n_keypoints = 4
    points = torch.randn((n_batch, n_views, n_keypoints, 2))

    points[0, 0, 0, :] = float('nan')  # nan1
    points[0, 0, 1, :] = float('nan')  # nan2
    points[1, 2, 3, :] = float('nan')  # nan3

    masks = get_valid_projection_masks(points)

    assert masks.shape == (n_batch, 3, n_keypoints)  # 3 = 3 choose 2

    # effect of nan1
    assert ~masks[0, 0, 0]
    masks[0, 0, 0] = True
    assert ~masks[0, 1, 0]
    masks[0, 1, 0] = True
    # effect of nan2
    assert ~masks[0, 0, 1]
    masks[0, 0, 1] = True
    assert ~masks[0, 1, 1]
    masks[0, 1, 1] = True
    # effect of nan3
    assert ~masks[1, 1, 3]
    masks[1, 1, 3] = True
    assert ~masks[1, 2, 3]
    masks[1, 2, 3] = True

    # test others
    assert torch.all(masks)
