import os


def test_get_model_folders(tmpdir):

    from lightning_pose.apps.utils import get_model_folders

    # create model directories
    models = {
        "00-11-22/33-44-55": {"predictions": True, "tb_logs": True},
        "11-22-33/44-55-66": {"predictions": True, "tb_logs": False},
        "22-33-44/55-66-77": {"predictions": False, "tb_logs": True},
        "33-44-55/66-77-88": {"predictions": False, "tb_logs": False},
    }
    model_parent = os.path.join(str(tmpdir), "models")
    for model, files in models.items():
        model_dir = os.path.join(model_parent, model)
        os.makedirs(model_dir)
        if files["predictions"]:
            os.mknod(os.path.join(model_dir, "predictions.csv"))
        if files["tb_logs"]:
            os.makedirs(os.path.join(model_dir, "tb_logs"))

    # test 1: find all model directories
    trained_models = get_model_folders(
        model_parent,
        require_predictions=False,
        require_tb_logs=False,
    )
    for model in models.keys():
        assert os.path.join(model_parent, model) in trained_models

    # test 2: find trained model directories with predictions.csv
    trained_models = get_model_folders(
        model_parent,
        require_predictions=True,
        require_tb_logs=False,
    )
    for model, files in models.items():
        if files["predictions"]:
            assert os.path.join(model_parent, model) in trained_models
        else:
            assert os.path.join(model_parent, model) not in trained_models

    # test 3: find trained model directories with config.yaml
    trained_models = get_model_folders(
        model_parent,
        require_predictions=False,
        require_tb_logs=True,
    )
    for model, files in models.items():
        if files["tb_logs"]:
            assert os.path.join(model_parent, model) in trained_models
        else:
            assert os.path.join(model_parent, model) not in trained_models

    # test 4: find trained model directories with both predictions.csv and config.yaml
    trained_models = get_model_folders(
        model_parent,
        require_predictions=True,
        require_tb_logs=True,
    )
    for model, files in models.items():
        if files["predictions"] and files["tb_logs"]:
            assert os.path.join(model_parent, model) in trained_models
        else:
            assert os.path.join(model_parent, model) not in trained_models
