Model Directory Structure
==========================

Project Schema v2.0.5

.. code-block:: text

   /models/model_dir/
   ├── config.yaml .......................................... [1]
   ├── predictions.csv ...................................... [2]
   ├── tb_logs/ ............................................. [3]
   │   ├── session0_view0/
   │   │   └── img001.png ................................ [4]
   │   └── session0_view1/
   │       └── img001.png
   ├── train_status/ ..................................... [3]
   ├── inference_status/ ..................................... [3]
   ├── video_preds/ ........................................... [5]
   │   ├── session0_view0.mp4/
   │   └── session0_view1.mp4/
   └── image_preds/ ........................................... [5]
       ├── session0_view0.mp4
       └── session0_view1.mp4

Breaking changes
-----------------

- video_preds_infer directory


1/14/2025 - v2.0.5

Deprecated:
- label files like `view0.csv`
- video_preds_infer directory in models

These are incompatible with the new app.
We will publish a migration guide from old app to new app soon.


The schema changes doc lists out all changes from version to version.

The directory structure reference fully specifies the latest schema.


Use it to identify what version of the schema you're on,

The "version" of the directory structure is the
lightning-pose release version where we introduced some directory structure change.

The change could be related to a feature you're not using,
in which case you don't have to do anything to be fully compatible with the new
lightning pose release.

See the schema changes docs for a full list of changes. The breaking changes
will be called out so you know whether its safe to upgrade lightning-pose.

We are working on better tooling to automate schema migrations.

So if your schema was correct as of 1.8.0




directory structures compatible with new lightning-pose releases

Older schema versions will have limited ongoing support, so it's important to
upgrade your directory structure as you upgrade lightning pose.
We are working on automating the schema migration process.




