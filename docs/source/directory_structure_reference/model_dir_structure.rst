Model Directory Structure
==========================

.. code-block:: text

   /models/model_dir/
   ├── config.yaml
   |
   # Singleview
   ├── predictions.csv
   # Multiview
   ├── predictions_view0.csv
   ├── predictions_view1.csv
   |
   ├── tb_logs/ ...
   ├── video_preds/
   │   ├── session0_view0.mp4/
   │   |   └── predictions.csv
   │   └── session0_view1.mp4/
   └── image_preds/
       └── CollectedData.csv/
            └── predictions.csv

    


