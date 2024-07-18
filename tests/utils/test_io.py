"""Test the io module."""

import os
import shutil

import pytest


def test_check_if_semisupervised():

    from lightning_pose.utils.io import check_if_semi_supervised

    flag = check_if_semi_supervised(losses_to_use=None)
    assert not flag

    flag = check_if_semi_supervised(losses_to_use=[])
    assert not flag

    flag = check_if_semi_supervised(losses_to_use=[""])
    assert not flag

    flag = check_if_semi_supervised(losses_to_use=["any_string"])
    assert flag

    flag = check_if_semi_supervised(losses_to_use=["loss1", "loss2"])
    assert flag


def test_get_videos_in_dir(toy_data_dir, tmpdir):

    from lightning_pose.utils.io import get_videos_in_dir

    videos_dir = os.path.join(toy_data_dir, "videos")

    # --------------------
    # single view tests
    # --------------------
    # test 1: single video
    video_list = get_videos_in_dir(videos_dir)
    assert len(video_list) == 1

    # test 2: two videos
    test_2_dir = os.path.join(str(tmpdir), "test_2")
    os.makedirs(test_2_dir)
    shutil.copyfile(video_list[0], os.path.join(test_2_dir, "vid1.mp4"))
    shutil.copyfile(video_list[0], os.path.join(test_2_dir, "vid2.mp4"))
    video_list_2 = get_videos_in_dir(test_2_dir)
    assert len(video_list_2) == 2

    # test 3: don't pick up non-video files
    shutil.copyfile(video_list[0], os.path.join(test_2_dir, "vid3.xyz"))
    video_list_3 = get_videos_in_dir(test_2_dir)
    assert len(video_list_3) == 2
    assert all([v1 == v2 for v1, v2 in zip(video_list_2, video_list_3)])

    # --------------------
    # multiview tests
    # --------------------
    view_names = ["top", "bot"]

    # test 4: single video for each view
    test_4_dir = os.path.join(str(tmpdir), "test_4")
    os.makedirs(test_4_dir)
    for view_name in view_names:
        shutil.copyfile(video_list[0], os.path.join(test_4_dir, f"vid1_{view_name}.mp4"))
    video_list_4 = get_videos_in_dir(test_4_dir, view_names=view_names)
    assert len(video_list_4) == len(view_names)
    for v_list in video_list_4:
        assert len(v_list) == 1

    # test 5: two videos for each view
    for view_name in view_names:
        shutil.copyfile(video_list[0], os.path.join(test_4_dir, f"vid2_{view_name}.mp4"))
    video_list_5 = get_videos_in_dir(test_4_dir, view_names=view_names)
    assert len(video_list_5) == len(view_names)
    for v_list in video_list_5:
        assert len(v_list) == 2

    # test 6: ignore a video if it doesn't match the provided views
    shutil.copyfile(video_list[0], os.path.join(test_4_dir, "vid2_view3.mp4"))
    video_list_6 = get_videos_in_dir(test_4_dir, view_names=view_names)
    assert len(video_list_6) == len(view_names)
    for v_list in video_list_6:
        assert len(v_list) == 2

    # test 7: fail if there is a mismatch in number of videos across views
    shutil.copyfile(video_list[0], os.path.join(test_4_dir, f"vid3_{view_names[0]}.mp4"))
    with pytest.raises(RuntimeError):
        get_videos_in_dir(test_4_dir, view_names=view_names)


def test_check_video_paths(toy_data_dir, tmpdir):

    from lightning_pose.utils.io import check_video_paths

    videos_dir = os.path.join(toy_data_dir, "videos")

    # --------------------
    # single view tests
    # --------------------
    # test 1: pass directory, single video
    video_list = check_video_paths(videos_dir)
    assert len(video_list) == 1

    # test 2: pass directory, two videos
    test_2_dir = os.path.join(str(tmpdir), "test_2")
    os.makedirs(test_2_dir)
    shutil.copyfile(video_list[0], os.path.join(test_2_dir, "vid1.mp4"))
    shutil.copyfile(video_list[0], os.path.join(test_2_dir, "vid2.mp4"))
    video_list_2 = check_video_paths(test_2_dir)
    assert len(video_list_2) == 2

    # test 3: pass list, two videos, should get the list back
    video_list_3 = check_video_paths(video_list_2)
    assert all([v1 == v2 for v1, v2 in zip(video_list_2, video_list_3)])

    # test 4: pass single file, get list with that file back
    video_list_4 = check_video_paths(video_list[0])
    assert len(video_list_4) == 1
    assert video_list_4[0] == video_list[0]

    # --------------------
    # multiview tests
    # --------------------
    # simple testing here, more thorough testing in test_get_videos_in_dir

    view_names = ["top", "bot"]

    # test 5: single video for each view
    test_5_dir = os.path.join(str(tmpdir), "test_5")
    os.makedirs(test_5_dir)
    for view_name in view_names:
        shutil.copyfile(video_list[0], os.path.join(test_5_dir, f"vid1_{view_name}.mp4"))
    video_list_5 = check_video_paths(test_5_dir, view_names=view_names)
    assert len(video_list_5) == len(view_names)
    for v_list in video_list_5:
        assert isinstance(v_list, list)
        assert len(v_list) == 1
