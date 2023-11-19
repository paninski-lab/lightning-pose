########################
Temporal Context Network
########################

The use of context frames is another core component of the Lightning Pose algorithm -
rather than predicting keypoints at time *t* solely from the frame at time *t*,
the Temporal Context Network (TCN) uses frames at times [*t-2*, *t-1*, *t*, *t+1*, *t+2*]
(but only requires labels for time *t*).
This temporal context can be especially helpful for resolving brief occlusions.
This page describes updates to the data and config file in order to properly use the TCN.

Data
====
The TCN requires the addition of context frames in the labeled data directory
(referred to as ``<LABELED_DATA_DIR>`` in :ref:`Organizing your data <directory_structure>`).

For example, if the labels csv file contains a frame named ``labeled-data/session_00/img009.png``
then you will need to add the frames ``img007.png``, ``img008.png``, ``img010.png``, ``img011.png``
to the directory ``labeled-data/session_00``. You *do not* need to change the labels csv file.

To extract specific frames from a video file, you can use the following python function:

.. code-block:: python

    import numpy as np

    def get_frames_from_idxs(cap, idxs):
        """Helper function to load video segments.

        Note
        ----
        To create the VideoCapture object:
        >>> import cv2
        >>> cap = cv2.VideoCapture(/path/to/video_file)

        Parameters
        ----------
        cap : cv2.VideoCapture object
        idxs : array-like
            frame indices into video

        Returns
        -------
        np.ndarray
            returned frames of shape shape (n_frames, n_channels, ypix, xpix)

        """
        is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
        n_frames = len(idxs)
        for fr, i in enumerate(idxs):
            if fr == 0 or not is_contiguous:
                cap.set(1, i)
            ret, frame = cap.read()
            if ret:
                if fr == 0:
                    height, width, _ = frame.shape
                    frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
                frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print(
                    'warning! reached end of video; returning blank frames for remainder of ' +
                    'requested indices')
                break
        return frames

.. warning::

    Check that this function returns the correct frames!
    Select a labeled frame that lives in the ``<LABELED_DATA_DIR>`` directory.
    Load that exact frame from the raw video and make sure the two match.

Config file
============

.. note::

    Recall that any of the config options can be updated directly from the command line;
    see the :ref:`Training <training>` section.

There is only one field of the config file that *must* be updated to properly fit the TCN model,
found in the ``model`` section:

.. code-block:: yaml

    model:
      model_type: heatmap_mhcrnn

Batch sizes
-----------

**Supervised training**:
The supervised TCN model requires 5x more memory than the standard supervised model, due to the
context frames. You might need to reduce the labeled batch size in ``training.train_batch_size`` to
avoid out of memory errors.

**Semi-supervised training**:
Context frames can be trivially combined with unsupervised losses to produce a semi-supervised
context model; all that is required is to set ``model.losses_to_use`` as described in the
:ref:`Unsupervised losses <unsupervised_losses>` section.
The semi-supervised context model requires at least 5x more memory than the supervised model,
depending on the unlabeled batch size.
The unlabeled batch size for the context model can be set with ``dali.context.train.batch_size``.

**Supervised/semi-supervised inference**:
Inference in the TCN model (supervised or unsupervised) is efficiently implemented so that each
frame in a sequence is only processed once; therefore you may not need to adjust inference
batch size, which is found at ``dali.context.predict.sequence_length``.
