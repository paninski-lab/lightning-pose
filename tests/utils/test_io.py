"""Test the io module."""

import os
import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from lightning_pose.utils import io as io_utils
from lightning_pose.utils.io import (
    check_if_semi_supervised,
    check_video_paths,
    collect_video_files_by_view,
    extract_session_name_from_video,
    find_video_files_for_views,
    get_context_img_paths,
    get_videos_in_dir,
)


def test_check_if_semisupervised():
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


def test_collect_video_files_by_view():
    # Simple case: files are in reverse order of the views.
    view_to_file = collect_video_files_by_view(
        [Path("a/simple_top.mp4"), Path("a/simple_bot.mp4")], ["bot", "top"]
    )
    assert view_to_file == {
        "bot": Path("a/simple_bot.mp4"),
        "top": Path("a/simple_top.mp4"),
    }

    # we just match based on view matching, we don't care about diffs in the rest of the string
    # TBD whether this is the desired behavior.
    view_to_file = collect_video_files_by_view(
        [Path("a/complex_top_2.mp4"), Path("a/simple_bot.mp4")], ["bot", "top"]
    )
    assert view_to_file == {
        "bot": Path("a/simple_bot.mp4"),
        "top": Path("a/complex_top_2.mp4"),
    }

    # We check the filename stem, not the entire path.
    view_to_file = collect_video_files_by_view(
        [Path("a/a.mp4"), Path("a/b.mp4")], ["b", "a"]
    )
    assert view_to_file == {
        "b": Path("a/b.mp4"),
        "a": Path("a/a.mp4"),
    }

    # View name must be separated by a delimiter.
    view_to_file = collect_video_files_by_view(
        [Path("a/aview_a.mp4"), Path("a/aview_b.mp4")], ["b", "a"]
    )
    assert view_to_file == {
        "b": Path("a/aview_b.mp4"),
        "a": Path("a/aview_a.mp4"),
    }


def test_get_context_img_paths():
    assert get_context_img_paths(Path("a/b/c/img_2.png")) == [
        Path("a/b/c/img_0.png"),
        Path("a/b/c/img_1.png"),
        Path("a/b/c/img_2.png"),
        Path("a/b/c/img_3.png"),
        Path("a/b/c/img_4.png"),
    ]

    assert get_context_img_paths(Path("a/b/c/img_0200.png")) == [
        Path("a/b/c/img_0198.png"),
        Path("a/b/c/img_0199.png"),
        Path("a/b/c/img_0200.png"),
        Path("a/b/c/img_0201.png"),
        Path("a/b/c/img_0202.png"),
    ]

    # Test negative indices floored to 0.
    assert get_context_img_paths(Path("a/b/c/img_1.png")) == [
        Path("a/b/c/img_0.png"),
        Path("a/b/c/img_0.png"),
        Path("a/b/c/img_1.png"),
        Path("a/b/c/img_2.png"),
        Path("a/b/c/img_3.png"),
    ]

    # Too many files.
    with pytest.raises(AssertionError):
        collect_video_files_by_view(
            [Path("foo.mp4"), Path("bar.mp4"), Path("baz.mp4")], ["foo", "bar"]
        )

    # Not enough files.
    with pytest.raises(AssertionError):
        collect_video_files_by_view(
            [Path("foo.mp4")], ["foo", "bar"])

    # two files for foo.
    with pytest.raises(ValueError):
        collect_video_files_by_view(
            [Path("foo1.mp4"), Path("foo2.mp4")], ["foo", "bar"])

    # file for bar not found.
    with pytest.raises(ValueError):
        collect_video_files_by_view(
            [Path("foo.mp4"), Path("baz.mp4")], ["foo", "bar"])


def test_extract_session_name_from_video():
    view_names = ["top", "bot"]

    # Test 1: basic case with view name
    session_name = extract_session_name_from_video("mouse_session_top.mp4", view_names)
    assert session_name == "mouse_session"

    # Test 2: basecase case with second view name
    session_name = extract_session_name_from_video("mouse_session_bot", view_names)
    assert session_name == "mouse_session"

    # Test 3: view name at the middle
    session_name = extract_session_name_from_video("mouse_session_top_135.mp4", view_names)
    assert session_name == "mouse_session_135"

    # Test 4: no view name in filename
    session_name = extract_session_name_from_video("mouse_session.mp4", view_names)
    assert session_name == "mouse_session"

    # Test 5: multiple underscores in filename + view name at the middle
    session_name = extract_session_name_from_video("mouse_session_2023_top_135.mp4", view_names)
    assert session_name == "mouse_session_2023_135"


def test_find_video_files_for_views(toy_data_dir, tmpdir):
    # Create test directory with video files
    test_dir = os.path.join(str(tmpdir), "test_videos")
    os.makedirs(test_dir)

    # Get the toy video file using the same method as other tests
    videos_dir = os.path.join(toy_data_dir, "videos")
    toy_video_list = get_videos_in_dir(videos_dir)
    toy_video = toy_video_list[0]

    # Create test video files
    video1_path = os.path.join(test_dir, "session1_top.mp4")
    video2_path = os.path.join(test_dir, "session1_bot.mp4")

    # Copy toy video to create test files
    shutil.copyfile(toy_video, video1_path)
    shutil.copyfile(toy_video, video2_path)

    view_names = ["top", "bot"]

    # Test 1: find videos for specified views
    video_files = find_video_files_for_views(test_dir, view_names)
    assert len(video_files) == 2
    assert "top" in video_files[0]
    assert "bot" in video_files[1]

    # Test 2: directory doesn't exist
    with pytest.raises(FileNotFoundError):
        find_video_files_for_views("/nonexistent/dir", view_names)

    # Test 3: no video files in directory
    empty_dir = os.path.join(str(tmpdir), "empty_dir")
    os.makedirs(empty_dir)
    with pytest.raises(FileNotFoundError):
        find_video_files_for_views(empty_dir, view_names)


def test_fix_empty_first_row(tmp_path):
    # Set up a df that we're trying to fix
    test_csv = textwrap.dedent(
        """
        scorer,test,test,test,test
        bodyparts,bp_1,bp_1,bp_2,bp_2
        coords,x,y,x,y
        labeled-data/test1.png,,,,
        labeled-data/test2.png,,,,
        labeled-data/test3.png,1.0,2.0,3.0,4.0
        """
    )
    test_csv_path = tmp_path / "test.csv"
    test_csv_path.write_text(test_csv)

    df = pd.read_csv(test_csv_path, header=[0, 1, 2], index_col=0)
    # Confirm this df has the problem.
    assert len(df) == 2
    assert df.index.name == "labeled-data/test1.png"
    assert df.index[0] == "labeled-data/test2.png"

    # Test the fix
    fixed_df = io_utils.fix_empty_first_row(df)
    assert len(fixed_df) == 3
    assert fixed_df.index.name is None
    assert fixed_df.index[0] == "labeled-data/test1.png"
