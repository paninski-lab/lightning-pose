"""Video file discovery, validation, and multiview filename parsing.

**Listing and guards** (single path in, files out):

- :func:`is_video_file`, :func:`list_video_files` — the shared definition of "a video file" and
  the one place directories are scanned; accepted formats are :data:`VIDEO_SUFFIXES`.
- :func:`has_videos`, :func:`require_videos` — check that a path contains videos.
- :func:`normalize_video_paths` — turn a file, list, or directory into a validated list of
  paths (this is what the semi-supervised data module and test-video prediction use).

**Multiview filename parsing** (a view name is a token in the filename; see
:func:`filename_matches_view`):

- :func:`extract_view_name_from_video`, :func:`extract_session_name_from_video` — parse one
  filename.
- :func:`match_videos_to_views` — one session's videos -> ``{view: path}``.
- :func:`group_videos_by_session` — many videos -> sessions x views. ``strict=False`` skips
  sessions missing a view with a warning; ``strict=True`` raises.
- :func:`get_videos_in_dir` — directory -> views x sessions (the layout the DALI readers want),
  using strict grouping.
"""

from __future__ import annotations  # python 3.8 compatibility for sphinx

import collections
import logging
import re
from pathlib import Path
from typing import overload

logger = logging.getLogger(__name__)

# video formats accepted for training and inference; compared case-insensitively
VIDEO_SUFFIXES: tuple[str, ...] = ('.mp4',)

# to ignore imports for sphinx-autoapidoc
__all__ = [
    'VIDEO_SUFFIXES',
    'extract_session_name_from_video',
    'extract_view_name_from_video',
    'filename_matches_view',
    'get_videos_in_dir',
    'group_videos_by_session',
    'has_videos',
    'is_video_file',
    'list_video_files',
    'match_videos_to_views',
    'normalize_video_paths',
    'require_videos',
]


# ---------------------------------------------------------------------------------------------
# Single-file primitives
# ---------------------------------------------------------------------------------------------

def is_video_file(path: str | Path, suffixes: tuple[str, ...] = VIDEO_SUFFIXES) -> bool:
    """Return whether ``path`` has a supported video suffix (case-insensitive).

    This only inspects the name; it does not check that the file exists.

    Args:
        path: path to test.
        suffixes: accepted suffixes, including the leading dot.

    Returns:
        ``True`` if the suffix is in ``suffixes``.
    """
    return Path(path).suffix.lower() in suffixes


def list_video_files(
    path: str | Path,
    suffixes: tuple[str, ...] = VIDEO_SUFFIXES,
) -> list[Path]:
    """List the video files at ``path``, which may be a single file or a directory.

    Args:
        path: a video file or a directory of videos (not searched recursively).
        suffixes: accepted suffixes, including the leading dot.

    Returns:
        sorted list of video files; ``[path]`` for a video file, and an empty list if the path
        is missing, is not a video file, or is a directory without videos.
    """
    path = Path(path)
    if path.is_file():
        return [path] if is_video_file(path, suffixes) else []
    if path.is_dir():
        return sorted(f for f in path.iterdir() if f.is_file() and is_video_file(f, suffixes))
    return []


# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------

def has_videos(video_dir: str | Path | None) -> bool:
    """Return whether ``video_dir`` is a video file or a directory containing videos.

    Args:
        video_dir: path to a video file or a directory of videos; may be ``None``.

    Returns:
        ``True`` if the path is an existing video file or a directory with at least one video;
        ``False`` otherwise.
    """
    return video_dir is not None and len(list_video_files(video_dir)) > 0


def require_videos(video_dir: str | Path | None, purpose: str) -> None:
    """Raise a descriptive error unless ``video_dir`` contains videos.

    Args:
        video_dir: path to a video file or a directory of videos; may be ``None``.
        purpose: what the videos are needed for, used in the error message.

    Raises:
        FileNotFoundError: if ``video_dir`` is unset, missing, or contains no videos.
    """
    if not has_videos(video_dir):
        raise FileNotFoundError(
            f'{purpose} requires a non-empty video directory; found {video_dir}.'
        )


# ---------------------------------------------------------------------------------------------
# Multiview filename parsing
# ---------------------------------------------------------------------------------------------

def filename_matches_view(filename: str | Path, view_name: str) -> bool:
    """Return whether a video filename belongs to a view.

    The view name must appear in the filename stem delimited by non-alphanumeric characters (or
    the start/end of the stem). For example, ``mouse_top_3.mp4`` matches view ``top``, but
    ``mousetop3.mp4`` does not, and ``sess_cam10.mp4`` matches ``cam10`` but not ``cam1``. This is
    the single matching rule used for all multiview video-filename parsing.

    Args:
        filename: video filename or path.
        view_name: name of the view.

    Returns:
        ``True`` if the filename contains the view name as a delimited token.
    """
    pattern = rf'(?<![0-9a-zA-Z]){re.escape(view_name)}(?![0-9a-zA-Z])'
    return re.search(pattern, Path(filename).stem) is not None


def extract_view_name_from_video(
    video_filename: str, view_names: list[str]
) -> str | None:
    """Return the view name that a video filename matches, or ``None`` if there is none.

    Args:
        video_filename: name of the video file (with or without extension)
        view_names: list of possible view names

    Returns:
        The first view in ``view_names`` that matches (see :func:`filename_matches_view`).
    """
    for view_name in view_names:
        if filename_matches_view(video_filename, view_name):
            return view_name
    return None


def extract_session_name_from_video(video_filename: str, view_names: list[str]) -> str:
    """Extract the session name from a video filename by removing the view name.

    Args:
        video_filename: name of the video file (with or without extension)
        view_names: list of possible view names to remove

    Returns:
        Session name with ``_<view_name>`` removed; the filename stem if no view name matches
        (see :func:`filename_matches_view`).
    """
    name_without_ext = Path(video_filename).stem

    view_name = extract_view_name_from_video(video_filename, view_names)
    if view_name is None:
        return name_without_ext

    session_name = name_without_ext.replace(f"_{view_name}", "")
    if session_name == name_without_ext:
        # view name was found but not in the expected "_<view>" form (e.g. a hyphen was used
        # instead); the session name below still contains the view name, so this video likely
        # won't be grouped with its other views
        logger.warning(
            f"video filename '{video_filename}' contains view name '{view_name}' but "
            f"not as a '_{view_name}' suffix; expected the pattern "
            f"'<session>_{view_name}'. This video may not be grouped correctly with "
            "its other views for multiview prediction."
        )
    return session_name


def match_videos_to_views(
    video_files: list[Path], view_names: list[str]
) -> dict[str, Path]:
    """Match the videos of a single session to views based on their filenames.

    Filenames must contain their corresponding view's name, separated from the rest of the
    filename by some non-alphanumeric delimiter (see :func:`filename_matches_view`). For example,
    mouse_top_3.mp4 is allowed, but mousetop3.mp4 is not allowed.

    Args:
        video_files: one video per view.
        view_names: names of the views.

    Returns:
        Mapping from view name to video file.

    Raises:
        ValueError: if a file matches multiple views, or no file is found for a view.
    """
    assert len(video_files) == len(view_names), (
        f"{len(video_files)} != {len(view_names)}"
    )
    video_files_by_view: dict[str, Path] = {}
    for view_name in view_names:
        # Search all the video_files for a match.
        for video_file in video_files:
            if filename_matches_view(video_file.name, view_name):
                if view_name not in video_files_by_view:
                    video_files_by_view[view_name] = video_file
                else:
                    raise ValueError(f"File matches multiple views: {video_file}")
        # After the search if nothing was added to dict, there is a problem.
        if view_name not in video_files_by_view:
            raise ValueError(f"File not found for view: {view_name}")

    return video_files_by_view


def group_videos_by_session(
    video_paths: list[Path],
    view_names: list[str],
    strict: bool = False,
) -> list[list[Path]]:
    """Group a list of videos from different sessions and views by session.

    Returns a list of lists like
    `[[sess0_view0.mp4, sess0_view1.mp4, ...], [sess1_view0.mp4, sess1_view1.mp4, ...], ...]`,
    i.e. sessions x views. Views are matched with :func:`filename_matches_view`.

    Args:
        video_paths: paths to video files from any number of sessions and views.
        view_names: names of the views; each returned session has one video per view, in this
            order.
        strict: how to handle sessions that are missing a view. If ``False`` (default), they are
            skipped with a warning naming the session and the missing views. If ``True``, a
            ``ValueError`` is raised instead. In both modes, videos that match none of the views
            are skipped with a warning.

    Returns:
        List for each complete session, each containing a sub-list with the video for each view.

    Raises:
        ValueError: if ``strict`` and a session is missing a view, or no video matches any view.
    """
    # map of session -> view -> video
    session_view_video_map: dict[str, dict[str, Path]] = collections.defaultdict(dict)
    unmatched: list[Path] = []

    for video_path in video_paths:
        view = extract_view_name_from_video(video_path.name, view_names)
        if view is None:
            unmatched.append(video_path)
            continue
        session = extract_session_name_from_video(video_path.name, view_names)
        session_view_video_map[session][view] = video_path

    if unmatched:
        logger.warning(
            f'skipping {len(unmatched)} video(s) that match none of the views {view_names}: '
            f'{[p.name for p in unmatched]}. Filenames must contain the view name delimited by '
            'non-alphanumeric characters, e.g. <session>_<view>.mp4.'
        )

    sessions: list[list[Path]] = []
    incomplete: dict[str, list[str]] = {}
    for session, view_to_video_path in session_view_video_map.items():
        missing = [view for view in view_names if view not in view_to_video_path]
        if missing:
            incomplete[session] = missing
        else:
            sessions.append([view_to_video_path[view] for view in view_names])

    if incomplete:
        details = '; '.join(f"'{s}' is missing {m}" for s, m in incomplete.items())
        if strict:
            raise ValueError(
                f'every session must have a video for every view {view_names}; {details}'
            )
        logger.warning(f'skipping {len(incomplete)} session(s) with missing views: {details}')
    elif strict and not sessions:
        raise ValueError(f'no videos match the views {view_names}')

    return sessions


# ---------------------------------------------------------------------------------------------
# Directory-level entry points
# ---------------------------------------------------------------------------------------------

@overload
def get_videos_in_dir(
    video_dir: str,
    view_names: None = None,
    return_mp4_only: bool = True,
) -> list[str]: ...


@overload
def get_videos_in_dir(
    video_dir: str,
    view_names: list[str],
    return_mp4_only: bool = True,
) -> list[list[str]]: ...


def get_videos_in_dir(
    video_dir: str, view_names: list[str] | None = None, return_mp4_only: bool = True
) -> list[str] | list[list[str]]:
    """Gather the videos in a directory as strings, laid out the way the DALI readers expect.

    For multiview, sessions are grouped with :func:`group_videos_by_session` in strict mode and
    transposed, so every view must have a video for every session.

    Args:
        video_dir: directory containing the videos.
        view_names: for multiview models, the view names; filenames must contain their view's
            name delimited by non-alphanumeric characters (see :func:`filename_matches_view`).
        return_mp4_only: if ``False``, also accept ``.avi`` and ``.mov`` files.

    Returns:
        Single view (``view_names`` is ``None``): flat list of video paths. Multiview: nested
        list with one inner list per view (in ``view_names`` order), each holding that view's
        videos across sessions, i.e. views x sessions.

    Raises:
        OSError: if the directory contains no videos.
        ValueError: if multiview and a session is missing a view (see
            :func:`group_videos_by_session`).
    """
    assert Path(video_dir).is_dir()
    suffixes = VIDEO_SUFFIXES if return_mp4_only else ('.mp4', '.avi', '.mov')
    video_files = list_video_files(video_dir, suffixes=suffixes)
    if len(video_files) == 0:
        raise OSError(f"Did not find any valid video files in {video_dir}")

    if not view_names:
        return [str(f) for f in video_files]

    sessions = group_videos_by_session(video_files, view_names, strict=True)
    # transpose sessions x views -> views x sessions
    return [[str(session[v]) for session in sessions] for v in range(len(view_names))]


@overload
def normalize_video_paths(
    video_paths: list[str] | str,
    view_names: None = None,
) -> list[str]: ...


@overload
def normalize_video_paths(
    video_paths: list[list[str]],
    view_names: None = None,
) -> list[list[str]]: ...


@overload
def normalize_video_paths(
    video_paths: list[str] | list[list[str]] | str,
    view_names: list[str],
) -> list[list[str]]: ...


def normalize_video_paths(
    video_paths: list[str] | list[list[str]] | str,
    view_names: list[str] | None = None,
) -> list[str] | list[list[str]]:
    """Validate and normalise video paths, returning a flat or nested list of mp4 paths.

    Accepts a single file path, a list of file paths, or a directory and returns a flat list of
    video paths for single-view models, or a nested list ``(views × sessions)`` for multi-view
    models.

    Args:
        video_paths: single video file path, list of video file paths (flat, or nested per view
            for multiview), or a directory.
        view_names: if provided, organise the paths by view name for a multi-view model.

    Returns:
        For single-view: ``list[str]`` of mp4 file paths.
        For multi-view: ``list[list[str]]`` where outer list indexes views.

    Raises:
        ValueError: if ``video_paths`` is not a valid file, list, or directory, or if any of the
            resulting files does not have a supported video suffix (see :data:`VIDEO_SUFFIXES`).
    """
    # get input data
    filenames: list[str] | list[list[str]]
    if isinstance(video_paths, list):
        # presumably a list of files
        filenames = video_paths
    elif isinstance(video_paths, str) and Path(video_paths).is_file():
        # single video file
        filenames = [video_paths]
    elif isinstance(video_paths, str) and Path(video_paths).is_dir():
        # directory of videos
        filenames = get_videos_in_dir(video_paths, view_names=view_names)
    else:
        raise ValueError(
            "`video_paths_list` must be a list of files, a single file, or a directory name"
        )
    unsupported = [
        f
        for filename in filenames
        for f in ([filename] if isinstance(filename, str) else filename)
        if not is_video_file(f)
    ]
    if unsupported:
        raise ValueError(
            f'video files must be one of {VIDEO_SUFFIXES} (h.264 codec); '
            f'unsupported files: {unsupported}'
        )

    return filenames
