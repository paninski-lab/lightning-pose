import typing


class ProjectKey(str):
    pass

class SessionKey(str):
    pass

class LabelFileKey(str):
    pass

class ViewName(str):
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