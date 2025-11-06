from abc import abstractmethod, ABC

from lightning_pose.utils.paths import ResourceType, AbstractResourceUtil


class BaseProjectSchemaV1(ABC):
    is_multiview: bool

    def __init__(self, is_multiview: bool):
        self.is_multiview = is_multiview

    @abstractmethod
    def for_(self, resource_type: ResourceType) -> AbstractResourceUtil:
        """Return the resource util for the given type."""
        raise NotImplementedError
