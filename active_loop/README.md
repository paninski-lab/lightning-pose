# Active Learning pipeline

Codebase:
- Make `iterations_folder`
- Given a run with a `config.yaml` and a `data` directory. The `data` directory has a `CollectedData.csv` file withe label information.
- Step 1: train a model using the config.yaml, which outputs a `predictions.csv` file
- Step 2: 
- [ ] Given a` predictions_new.csv file` select N frames at random <>
  - AL methods: Fix videos which you want to label.
        random sampling: input: predictions file: output: `random_frames.csv` .Possibly multiple runs, possibly data from multiple models.
        margin sampling: input: callback when creating predictions file that computes the margins in the heatmap and output: `margin_frames.csv`
        ensemble sampling: input: multiple predictions file (from different models), output: `ensemble_frames.csv`
        error-loss based sampling: input predictions file, outputs: `loss_frames.csv` (reprojection error, smoothing error)
- [ ] How to combine the frames from different methods? 
- [ ] create a new directory `active_run_$number_run$_method_$method$` with those `new_frames (and their labels)*` (labels available in debug mode or from user).
- [ ] make  `new_CollectedData.csv` which includes the the new frames and labels*.
- [ ] merge `new_CollectedData.csv` with `CollectedData.csv` to create a new `CollectedData.csv` file.
- [ ] create a `new config.yaml` file which points to the updated `CollectedData.csv`.


# Example:

- loop_iteration(method, data, loop_number)
  - if loop_number = 0
    - make data/iterations_folder
  - if loop_number > 0
    - copy `CollectedData.csv` (keep track)
    - run `select_frames(method)` on data.
      - output `iteration_#/'selected_frames/$method.csv'` rank on each frame.
      - call function `select_frames('all_methods')`: picks N frames from all methods.
      - make folder labeled_videos/frames (queryuser*)
      - make `CollectedData.csv` inside of iteration folder and move to data/

- Launch experiment:
- [ ] Step 0: start with folder with videos: `data/` 
  - split labeled data and unlabeled data into train/val/test + test_across_loop_iterations.
    - `Collected.csv` labeled: ibl1/Collected.csv 1 video with 1k labels (to train model)
    - `Collected_new.csv` unlabeled-videos: ibl1_corruption_level (to eval model, used to select frames)
    - `Collected_test_loop.csv`test_across_loop_iterations: not in the bucket to be labeled (ibl1_gaussian_noise_5, ibl1_brightness_5) (to compare across active_loop iterations) 
- [ ] Step 1: Select initial frames to label
  - loop_iteration_#(method='random', data=unlabeled videos)
- [ ] Step 2: Train a model:
  - this produces an outputs/#/#/ with `predictions.csv`, `predictions_new.csv`, `predictions_test_loop.csv`checkpoints,etc.
  - `CollectedData.csv`  = loop_iteration_#(method='random', data=`predictions.csv`)
  - GO back to step 1.

