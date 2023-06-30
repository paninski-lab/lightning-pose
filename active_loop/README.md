# Active Learning pipeline

Codebase:
- Make `iterations_folder`
- Given a run with a `config.yaml` and a `data` directory. The `data` directory has a `CollectedData.csv` file withe label information.
- Step 1: train a model using the config.yaml, which outputs a `predictions.csv` file
- Step 2: 
- [ ] Given a` predictions.csv file` select N frames at random <>
  - AL methods: Fix videos which you want to label.
        random sampling: input: predictions.csv: output: `random_frames.csv` .Possibly multiple runs, possibly data from multiple models.
        margin sampling: input: callback when creating predictions.csv that computes the margins in the heatmap and output: `margin_frames.csv`
        ensemble sampling: input: multiple predictions.csv files (from different models), output: `ensemble_frames.csv`
        error-loss based sampling: input predictions.csv, outputs: `loss_frames.csv` (reprojection error, smoothing error)
- [ ] How to combine the frames from different methods? 
- [ ] create a new directory `active_run_$number_run$_method_$method$` with those `new_frames (and their labels)*` (labels available in debug mode or from user).
- [ ] make  `new_CollectedData.csv` which includes the the new frames and labels.
- [ ] merge `new_CollectedData.csv` with `CollectedData.csv` to create a new `CollectedData.csv` file.
- [ ] create a `new config.yaml` file which points to the updated `CollectedData.csv`.
