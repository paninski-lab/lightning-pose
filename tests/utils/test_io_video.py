"""Tests for lightning_pose.utils.io_video."""

import logging
import os
import shutil
from pathlib import Path

import pytest

from lightning_pose.utils.io_video import (
    extract_session_name_from_video,
    extract_view_name_from_video,
    filename_matches_view,
    get_videos_in_dir,
    group_videos_by_session,
    has_videos,
    is_video_file,
    list_video_files,
    match_videos_to_views,
    normalize_video_paths,
    require_videos,
)


class TestIsVideoFile:
    """Test the function is_video_file."""

    @pytest.mark.parametrize('name', ['a.mp4', 'a.MP4', 'dir/a.Mp4'])
    def test_is_video_file_true(self, name):
        assert is_video_file(name)

    @pytest.mark.parametrize('name', ['a.avi', 'a.csv', 'a', 'mp4'])
    def test_is_video_file_false(self, name):
        assert not is_video_file(name)


class TestListVideoFiles:
    """Test the function list_video_files."""

    def test_list_video_files_directory_sorted_and_filtered(self, tmp_path):
        for name in ['b.mp4', 'a.MP4', 'c.avi', 'notes.txt']:
            (tmp_path / name).touch()
        (tmp_path / 'sub.mp4').mkdir()  # a directory with a video-like name is ignored
        assert list_video_files(tmp_path) == [tmp_path / 'a.MP4', tmp_path / 'b.mp4']

    def test_list_video_files_single_file(self, tmp_path):
        (tmp_path / 'a.mp4').touch()
        assert list_video_files(str(tmp_path / 'a.mp4')) == [tmp_path / 'a.mp4']

    def test_list_video_files_non_video_file(self, tmp_path):
        (tmp_path / 'a.csv').touch()
        assert list_video_files(tmp_path / 'a.csv') == []

    def test_list_video_files_missing_path(self, tmp_path):
        assert list_video_files(tmp_path / 'missing') == []

    def test_list_video_files_not_recursive(self, tmp_path):
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'sub' / 'a.mp4').touch()
        assert list_video_files(tmp_path) == []


class TestFilenameMatchesView:
    """Test the function filename_matches_view."""

    @pytest.mark.parametrize('name', ['mouse_top_3.mp4', 'top.mp4', 'a-top.mp4', 'mouse_top'])
    def test_filename_matches_view_true(self, name):
        assert filename_matches_view(name, 'top')

    @pytest.mark.parametrize(
        'name', ['mousetop.mp4', 'mousetop3.mp4', 'topdown.mp4', 'a_stop.mp4'],
    )
    def test_filename_matches_view_false(self, name):
        assert not filename_matches_view(name, 'top')

    def test_filename_matches_view_prefix_view_names_are_distinct(self):
        assert filename_matches_view('sess_cam10.mp4', 'cam10')
        assert not filename_matches_view('sess_cam10.mp4', 'cam1')

    def test_filename_matches_view_ignores_directory(self):
        assert not filename_matches_view('/data/top/vid.mp4', 'top')


class TestMatchVideosToViews:
    """Test the function match_videos_to_views."""

    def test_match_videos_to_views_requires_left_boundary(self):
        with pytest.raises(ValueError, match='File not found for view: top'):
            match_videos_to_views([Path('mousetop.mp4'), Path('mouse_bot.mp4')], ['top', 'bot'])

    def test_match_videos_to_views_ok(self):
        files = [Path('m_bot.mp4'), Path('m_top.mp4')]
        assert match_videos_to_views(files, ['top', 'bot']) == {
            'top': Path('m_top.mp4'),
            'bot': Path('m_bot.mp4'),
        }


class TestExtractViewAndSessionNames:
    """Test extract_view_name_from_video and extract_session_name_from_video."""

    def test_extract_view_name_prefix_view_names(self):
        assert extract_view_name_from_video('sess_cam10.mp4', ['cam1', 'cam10']) == 'cam10'
        assert extract_view_name_from_video('sess_cam1.mp4', ['cam1', 'cam10']) == 'cam1'

    def test_extract_view_name_none(self):
        assert extract_view_name_from_video('sess.mp4', ['top']) is None

    def test_extract_session_name_prefix_view_names(self):
        views = ['cam1', 'cam10']
        assert extract_session_name_from_video('sess0_cam10.mp4', views) == 'sess0'
        assert extract_session_name_from_video('sess0_cam1.mp4', views) == 'sess0'

    def test_extract_session_name_no_view_returns_stem(self):
        assert extract_session_name_from_video('sess0.mp4', ['top']) == 'sess0'

    def test_extract_session_name_warns_on_wrong_delimiter(self, caplog):
        with caplog.at_level(logging.WARNING):
            name = extract_session_name_from_video('sess0-top.mp4', ['top'])
        assert name == 'sess0-top'
        assert 'not as a' in caplog.text


class TestHasVideos:
    """Test the function has_videos."""

    def test_has_videos_directory_with_mp4(self, tmp_path):
        (tmp_path / 'vid.mp4').touch()
        assert has_videos(tmp_path)
        assert has_videos(str(tmp_path))

    def test_has_videos_single_file(self, tmp_path):
        (tmp_path / 'vid.mp4').touch()
        assert has_videos(tmp_path / 'vid.mp4')

    def test_has_videos_empty_directory(self, tmp_path):
        assert not has_videos(tmp_path)

    def test_has_videos_no_mp4_files(self, tmp_path):
        (tmp_path / 'notes.txt').touch()
        assert not has_videos(tmp_path)

    def test_has_videos_missing_path(self, tmp_path):
        assert not has_videos(tmp_path / 'missing')

    def test_has_videos_none(self):
        assert not has_videos(None)


class TestRequireVideos:
    """Test the function require_videos."""

    def test_require_videos_passes_with_videos(self, tmp_path):
        (tmp_path / 'vid.mp4').touch()
        require_videos(tmp_path, purpose='semi-supervised training')

    def test_require_videos_raises_for_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='semi-supervised training requires'):
            require_videos(tmp_path / 'missing', purpose='semi-supervised training')

    def test_require_videos_raises_for_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='non-empty video directory'):
            require_videos(tmp_path, purpose='semi-supervised training')

    def test_require_videos_raises_for_none(self):
        with pytest.raises(FileNotFoundError):
            require_videos(None, purpose='semi-supervised training')


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
    assert all([v1 == v2 for v1, v2 in zip(video_list_2, video_list_3, strict=True)])

    # --------------------
    # multiview tests
    # --------------------
    view_names = ["top", "bot"]

    # test 4: single video for each view
    test_4_dir = os.path.join(str(tmpdir), "test_4")
    os.makedirs(test_4_dir)
    for view_name in view_names:
        shutil.copyfile(
            video_list[0], os.path.join(test_4_dir, f"vid1_{view_name}.mp4")
        )
    video_list_4 = get_videos_in_dir(test_4_dir, view_names=view_names)
    assert len(video_list_4) == len(view_names)
    for v_list in video_list_4:
        assert len(v_list) == 1

    # test 5: two videos for each view
    for view_name in view_names:
        shutil.copyfile(
            video_list[0], os.path.join(test_4_dir, f"vid2_{view_name}.mp4")
        )
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
    shutil.copyfile(
        video_list[0], os.path.join(test_4_dir, f"vid3_{view_names[0]}.mp4")
    )
    with pytest.raises(ValueError, match='vid3.*missing'):
        get_videos_in_dir(test_4_dir, view_names=view_names)


class TestNormalizeVideoPathsSuffix:
    """Test suffix validation in the function normalize_video_paths."""

    def test_normalize_video_paths_unsupported_suffix_raises(self, tmp_path):
        (tmp_path / 'a.avi').touch()
        with pytest.raises(ValueError, match=r'unsupported files.*a\.avi'):
            normalize_video_paths([str(tmp_path / 'a.avi')])

    def test_normalize_video_paths_unsupported_suffix_multiview_lists_offenders(self, tmp_path):
        files = [[str(tmp_path / 'a_v0.mp4')], [str(tmp_path / 'a_v1.mov')]]
        with pytest.raises(ValueError, match=r'a_v1\.mov'):
            normalize_video_paths(files)

    def test_normalize_video_paths_uppercase_suffix_ok(self, tmp_path):
        (tmp_path / 'a.MP4').touch()
        assert normalize_video_paths(str(tmp_path)) == [str(tmp_path / 'a.MP4')]


def test_normalize_video_paths(toy_data_dir, tmpdir):
    videos_dir = os.path.join(toy_data_dir, "videos")

    # --------------------
    # single view tests
    # --------------------
    # test 1: pass directory, single video
    video_list = normalize_video_paths(videos_dir)
    assert len(video_list) == 1

    # test 2: pass directory, two videos
    test_2_dir = os.path.join(str(tmpdir), "test_2")
    os.makedirs(test_2_dir)
    shutil.copyfile(video_list[0], os.path.join(test_2_dir, "vid1.mp4"))
    shutil.copyfile(video_list[0], os.path.join(test_2_dir, "vid2.mp4"))
    video_list_2 = normalize_video_paths(test_2_dir)
    assert len(video_list_2) == 2

    # test 3: pass list, two videos, should get the list back
    video_list_3 = normalize_video_paths(video_list_2)
    assert all([v1 == v2 for v1, v2 in zip(video_list_2, video_list_3, strict=True)])

    # test 4: pass single file, get list with that file back
    video_list_4 = normalize_video_paths(video_list[0])
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
        shutil.copyfile(
            video_list[0], os.path.join(test_5_dir, f"vid1_{view_name}.mp4")
        )
    video_list_5 = normalize_video_paths(test_5_dir, view_names=view_names)
    assert len(video_list_5) == len(view_names)
    for v_list in video_list_5:
        assert isinstance(v_list, list)
        assert len(v_list) == 1


def test_match_videos_to_views():
    # Simple case: files are in reverse order of the views.
    view_to_file = match_videos_to_views(
        [Path("a/simple_top.mp4"), Path("a/simple_bot.mp4")], ["bot", "top"]
    )
    assert view_to_file == {
        "bot": Path("a/simple_bot.mp4"),
        "top": Path("a/simple_top.mp4"),
    }

    # we just match based on view matching, we don't care about diffs in the rest of the string
    # TBD whether this is the desired behavior.
    view_to_file = match_videos_to_views(
        [Path("a/complex_top_2.mp4"), Path("a/simple_bot.mp4")], ["bot", "top"]
    )
    assert view_to_file == {
        "bot": Path("a/simple_bot.mp4"),
        "top": Path("a/complex_top_2.mp4"),
    }

    # We check the filename stem, not the entire path.
    view_to_file = match_videos_to_views(
        [Path("a/a.mp4"), Path("a/b.mp4")], ["b", "a"]
    )
    assert view_to_file == {
        "b": Path("a/b.mp4"),
        "a": Path("a/a.mp4"),
    }

    # View name must be separated by a delimiter.
    view_to_file = match_videos_to_views(
        [Path("a/aview_a.mp4"), Path("a/aview_b.mp4")], ["b", "a"]
    )
    assert view_to_file == {
        "b": Path("a/aview_b.mp4"),
        "a": Path("a/aview_a.mp4"),
    }


def test_extract_session_name_from_video():
    view_names = ["top", "bot"]

    # Test 1: basic case with view name
    session_name = extract_session_name_from_video("mouse_session_top.mp4", view_names)
    assert session_name == "mouse_session"

    # Test 2: basecase case with second view name
    session_name = extract_session_name_from_video("mouse_session_bot", view_names)
    assert session_name == "mouse_session"

    # Test 3: view name at the middle
    session_name = extract_session_name_from_video(
        "mouse_session_top_135.mp4", view_names
    )
    assert session_name == "mouse_session_135"

    # Test 4: no view name in filename
    session_name = extract_session_name_from_video("mouse_session.mp4", view_names)
    assert session_name == "mouse_session"

    # Test 5: multiple underscores in filename + view name at the middle
    session_name = extract_session_name_from_video(
        "mouse_session_2023_top_135.mp4", view_names
    )
    assert session_name == "mouse_session_2023_135"


def test_extract_session_name_from_video_warns_on_non_underscore_separator(caplog):
    """View name found but not as a '_<view>' suffix (e.g. hyphen-separated): a warning is
    logged since the returned session name still contains the view name."""
    view_names = ["right", "left"]

    with caplog.at_level(logging.WARNING, logger="lightning_pose"):
        session_name = extract_session_name_from_video("recording-right.mp4", view_names)

    assert session_name == "recording-right"
    assert any(
        "not as a '_right' suffix" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


class TestGetVideosInDirMultiview:
    """Test the strict multiview layout of get_videos_in_dir."""

    def test_get_videos_in_dir_views_by_sessions_layout(self, tmp_path):
        for name in ['s1_top', 's1_bot', 's2_top', 's2_bot']:
            (tmp_path / f'{name}.mp4').touch()
        result = get_videos_in_dir(str(tmp_path), view_names=['top', 'bot'])
        assert result == [
            [str(tmp_path / 's1_top.mp4'), str(tmp_path / 's2_top.mp4')],
            [str(tmp_path / 's1_bot.mp4'), str(tmp_path / 's2_bot.mp4')],
        ]

    def test_get_videos_in_dir_view_without_videos_raises(self, tmp_path):
        (tmp_path / 's1_top.mp4').touch()
        with pytest.raises(ValueError, match=r"'s1' is missing \['bot'\]"):
            get_videos_in_dir(str(tmp_path), view_names=['top', 'bot'])

    def test_get_videos_in_dir_empty_raises(self, tmp_path):
        with pytest.raises(OSError, match='Did not find any valid video files'):
            get_videos_in_dir(str(tmp_path), view_names=['top', 'bot'])

    def test_get_videos_in_dir_accepts_avi_when_not_mp4_only(self, tmp_path):
        (tmp_path / 'a.avi').touch()
        assert get_videos_in_dir(str(tmp_path), return_mp4_only=False) == [
            str(tmp_path / 'a.avi')
        ]
        with pytest.raises(OSError):
            get_videos_in_dir(str(tmp_path))


class TestGroupVideosBySessionStrictness:
    """Test the strict flag and warnings of group_videos_by_session."""

    views = ['top', 'bot']

    def _paths(self, *names):
        return [Path(f'/d/{n}.mp4') for n in names]

    def test_lenient_warns_about_incomplete_session(self, caplog):
        paths = self._paths('s1_top', 's1_bot', 's2_top')
        with caplog.at_level(logging.WARNING):
            sessions = group_videos_by_session(paths, self.views)
        assert sessions == [[Path('/d/s1_top.mp4'), Path('/d/s1_bot.mp4')]]
        assert "'s2' is missing ['bot']" in caplog.text

    def test_lenient_warns_about_unmatched_files(self, caplog):
        paths = self._paths('s1_top', 's1_bot', 's1_vew1')
        with caplog.at_level(logging.WARNING):
            sessions = group_videos_by_session(paths, self.views)
        assert len(sessions) == 1
        assert 's1_vew1.mp4' in caplog.text
        assert 'match none of the views' in caplog.text

    def test_lenient_no_warning_when_all_complete(self, caplog):
        with caplog.at_level(logging.WARNING):
            group_videos_by_session(self._paths('s1_top', 's1_bot'), self.views)
        assert caplog.text == ''

    def test_strict_raises_on_incomplete_session(self):
        paths = self._paths('s1_top', 's1_bot', 's2_top')
        with pytest.raises(ValueError, match=r"'s2' is missing \['bot'\]"):
            group_videos_by_session(paths, self.views, strict=True)

    def test_strict_raises_when_nothing_matches(self):
        with pytest.raises(ValueError, match='no videos match'):
            group_videos_by_session(self._paths('s1_side'), self.views, strict=True)

    def test_strict_still_warns_about_unmatched_files(self, caplog):
        paths = self._paths('s1_top', 's1_bot', 's1_side')
        with caplog.at_level(logging.WARNING):
            sessions = group_videos_by_session(paths, self.views, strict=True)
        assert len(sessions) == 1
        assert 's1_side.mp4' in caplog.text


def test_group_videos_by_session():
    view_names = ["top", "bot"]

    # Test 1: one session with two views
    video_files_per_session = group_videos_by_session(
        [Path("/video_dir/session1_top"), Path("/video_dir/session1_bot")], view_names
    )
    assert len(video_files_per_session) == 1

    video_files = video_files_per_session[0]
    assert len(video_files) == 2
    assert "top" in str(video_files[0])
    assert "bot" in str(video_files[1])

    # Test 2: two sessions with two views
    video_files_per_session = group_videos_by_session(
        [
            Path("/video_dir/session1_top"),
            Path("/video_dir/session1_bot"),
            Path("/video_dir/session2_top"),
            Path("/video_dir/session2_bot"),
        ],
        view_names,
    )
    assert len(video_files_per_session) == 2

    video_files1 = video_files_per_session[0]
    assert len(video_files1) == 2
    assert "top" in str(video_files1[0])
    assert "bot" in str(video_files1[1])
    video_files2 = video_files_per_session[1]
    assert len(video_files2) == 2
    assert "top" in str(video_files2[0])
    assert "bot" in str(video_files2[1])

    # Test 3: extra unrelated video
    video_files_per_session = group_videos_by_session(
        [
            Path("/video_dir/session1_top"),
            Path("/video_dir/session1_bot"),
            Path("/video_dir/session1_side"),
        ],
        view_names,
    )
    assert len(video_files_per_session) == 1

    video_files = video_files_per_session[0]
    assert len(video_files) == 2
    assert "top" in str(video_files[0])
    assert "bot" in str(video_files[1])

    # Test 4: no relevant video files
    video_files_per_session = group_videos_by_session(
        [Path("/video_dir/session1_side")], view_names
    )
    assert len(video_files_per_session) == 0

    # Test 5: no video files
    video_files_per_session = group_videos_by_session([], view_names)
    assert len(video_files_per_session) == 0


def test_match_videos_to_views_errors():
    # Too many files.
    with pytest.raises(AssertionError):
        match_videos_to_views(
            [Path("foo.mp4"), Path("bar.mp4"), Path("baz.mp4")], ["foo", "bar"]
        )

    # Not enough files.
    with pytest.raises(AssertionError):
        match_videos_to_views([Path("foo.mp4")], ["foo", "bar"])

    # two files for foo.
    with pytest.raises(ValueError):
        match_videos_to_views(
            [Path("foo1.mp4"), Path("foo2.mp4")], ["foo", "bar"]
        )

    # file for bar not found.
    with pytest.raises(ValueError):
        match_videos_to_views([Path("foo.mp4"), Path("baz.mp4")], ["foo", "bar"])
