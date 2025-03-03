import warnings

from lightning_pose.apps.fiftyone import *  # Import the new module

# Emit a DeprecationWarning
warnings.warn(
    """
    The `lightning_pose.utils.fiftyone` module will be deprecated in a future release.
    Replace:
        import lightning_pose.utils.fiftyone
    with:
        import lightning_pose.apps.fiftyone
    """,
    DeprecationWarning,
)
