
lightning run app app.py --env NVIDIA_DRIVER_CAPABILITIES=all --cloud

lightning connect 01gm32h35gdy49gwz6769yfggb --yes


lightning run sweep --requirements=requirements.txt --packages="ffmpeg libsm6 libxext6" --syntax=hydra --pip-install-source --artifacts_path="." --cloud_compute=gpu-fast train_hydra.py training.train_frames=75,1 model.losses_to_use="[pca_singleview]","[temporal]"

lightning run experiment --requirements=requirements.txt --packages="ffmpeg libsm6 libxext6" --pip-install-source --artifacts_path="." --cloud_compute=gpu-fast train_hydra.py training.train_frames=75 model.losses_to_use="[pca_singleview]" training.min_epochs=10 training.max_epochs=30
