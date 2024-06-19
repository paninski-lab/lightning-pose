import copy
import os


def test_train(cfg, tmpdir):

    from lightning_pose.train import train

    pwd = os.getcwd()

    # copy config and update paths
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.data.data_dir = os.path.join(pwd, cfg_tmp.data.data_dir)
    cfg_tmp.data.video_dir = os.path.join(cfg_tmp.data.data_dir, "videos")
    cfg_tmp.eval.test_videos_directory = cfg_tmp.data.video_dir

    # don't train for long
    cfg_tmp.training.min_epochs = 2
    cfg_tmp.training.max_epochs = 2
    cfg_tmp.training.check_val_every_n_epoch = 1
    cfg_tmp.training.log_every_n_steps = 1
    cfg_tmp.training.limit_train_batches = 2

    # train simple model
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []

    # predict on vid
    cfg_tmp.eval.predict_vids_after_training = True
    cfg_tmp.eval.save_vids_after_training = True

    # change directory to save outputs elsewhere
    os.chdir(tmpdir)

    # train model
    train(cfg_tmp)

    # change directory back
    os.chdir(pwd)

    # ensure labeled data was properly processed
    assert os.path.isfile(os.path.join(tmpdir, "config.yaml"))
    assert os.path.isfile(os.path.join(tmpdir, "predictions.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "predictions_pca_multiview_error.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "predictions_pca_singleview_error.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "predictions_pixel_error.csv"))

    # ensure video data was properly processed
    assert os.path.isfile(os.path.join(tmpdir, "video_preds", "test_vid.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "video_preds", "test_vid_pca_multiview_error.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "video_preds", "test_vid_pca_singleview_error.csv"))
    assert os.path.isfile(os.path.join(tmpdir, "video_preds", "test_vid_temporal_norm.csv"))
    assert os.path.isfile(os.path.join(
        tmpdir, "video_preds", "labeled_videos", "test_vid_labeled.mp4",
    ))


def test_train_multiview(cfg_multiview, tmpdir):

    from lightning_pose.train import train

    pwd = os.getcwd()

    # copy config and update paths
    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.data.data_dir = os.path.join(pwd, cfg_tmp.data.data_dir)
    cfg_tmp.data.video_dir = os.path.join(cfg_tmp.data.data_dir, "videos")
    cfg_tmp.eval.test_videos_directory = cfg_tmp.data.video_dir

    # don't train for long
    cfg_tmp.training.min_epochs = 2
    cfg_tmp.training.max_epochs = 2
    cfg_tmp.training.check_val_every_n_epoch = 1
    cfg_tmp.training.log_every_n_steps = 1
    cfg_tmp.training.limit_train_batches = 2

    # train simple model
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []

    # predict on vid
    cfg_tmp.eval.predict_vids_after_training = True
    cfg_tmp.eval.save_vids_after_training = True

    # change directory to save outputs elsewhere
    os.chdir(tmpdir)

    # train model
    train(cfg_tmp)

    # change directory back
    os.chdir(pwd)

    assert os.path.isfile(os.path.join(tmpdir, "config.yaml"))

    for view in ["top", "bot"]:

        # ensure labeled data was properly processed
        assert os.path.isfile(os.path.join(tmpdir, f"predictions_{view}.csv"))
        assert os.path.isfile(os.path.join(tmpdir, f"predictions_{view}_pixel_error.csv"))
        # assert os.path.isfile(os.path.join(tmpdir, f"predictions_{view}_pca_multiview_error.csv"))
        assert os.path.isfile(os.path.join(tmpdir, f"predictions_{view}_pca_singleview_error.csv"))

        # ensure video data was properly processed
        assert os.path.isfile(os.path.join(
            tmpdir, "video_preds", f"test_vid_{view}.csv"))
        assert os.path.isfile(os.path.join(
            tmpdir, "video_preds", f"test_vid_{view}_temporal_norm.csv"))
        # assert os.path.isfile(os.path.join(
        #     tmpdir, "video_preds", f"test_vid_{view}_pca_multiview_error.csv"))
        assert os.path.isfile(os.path.join(
            tmpdir, "video_preds", f"test_vid_{view}_pca_singleview_error.csv"))
        assert os.path.isfile(os.path.join(
            tmpdir, "video_preds", "labeled_videos", f"test_vid_{view}_labeled.mp4"))
