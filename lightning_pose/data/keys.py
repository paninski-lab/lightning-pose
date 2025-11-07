import typing


class ProjectKey(str):
    """The key for a Project (aka Project Name)."""
    pass

class SessionKey(str):
    """The key for a Session (a multiview video)."""
    pass

class LabelFileKey(str):
    """The key for a multiview set of label files."""
    pass

class ViewName(str):
    """
    Trivial sublcass of str to make function arguments
    and return types a bit more clear.

    Also allows migration scripts to recognize the type of Key
    for type-specific sanitization.
    """
    pass


class VideoFileKey(typing.NamedTuple):
    session_key: SessionKey
    view: ViewName | None = None
    """
    def __repr__(self):
        return f"session={self.session_key},view={self.view}"
    """

class FrameKey(typing.NamedTuple):
    session_key: SessionKey
    frame_index: int
    view: ViewName | None = None
    """
    def __repr__(self):
        return f"session={self.session_key},view={self.view},frame_index={self.frame_index}"
"""