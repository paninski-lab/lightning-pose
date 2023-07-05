# Active Learning pipeline

Codebase:
- [x] Define active loop config, example in `configs/config_ibl_active.yaml`
- [x] Make `iterations` folder which is set in the config file.
- [x] Given a run with a `config.yaml` and a `data` directory. The `data` directory has a `CollectedData.csv` file withe label information.
- [x] Step 1: train a model using the config.yaml, which outputs a `predictions.csv` file
- Step 2: 
- [x] Given a` predictions_new.csv file` select N frames at random <>
  - AL methods: Fix videos which you want to label.
        random sampling: input: predictions file: output: `random_frames.csv` .Possibly multiple runs, possibly data from multiple models.
        margin sampling: input: callback when creating predictions file that computes the margins in the heatmap and output: `margin_frames.csv`
        ensemble sampling: input: multiple predictions file (from different models), output: `ensemble_frames.csv`
        error-loss based sampling: input predictions file, outputs: `loss_frames.csv` (reprojection error, smoothing error)
- [x] How to combine the frames from different methods? 
- [x] create a new file `iteration_active_loop/experiment0/${method}_${num_frames}` with the `new_frames (and their labels)*` (labels available in debug mode or from user).
- [x] merge previous run train frames in `CollectedData.csv` in new `${method}_${num_frames}_CollectedData.csv`  file.
- [x] update parameters in exp. config file (for example `configs/config_ibl_experiment.yaml`) to point to the updated `${method}_${num_frames}_CollectedData.csv`  file.


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
- [x] Step 0: start with folder with videos: `data/` 
  - split labeled data and unlabeled data into train/val/test + test_across_loop_iterations.
    - `Collected.csv` labeled: ibl1/Collected.csv 1 video with 1k labels (to train model)
    - `Collected_new.csv` unlabeled-videos: ibl1_corruption_level (to eval model, used to select frames)
    - `Collected_test_loop.csv`test_across_loop_iterations: not in the bucket to be labeled (ibl1_gaussian_noise_5, ibl1_brightness_5) (to compare across active_loop iterations) 
- [x] Step 1: Select initial frames to label
  - loop_iteration(method='random', data=unlabeled videos)
- [x] Step 2: Train a model:
  - this produces an outputs/#/#/ with `predictions.csv`, `predictions_new.csv`, `predictions_test_loop.csv`checkpoints,etc.
  - `CollectedData.csv`  = loop_iteration_#(method='random', data=`predictions.csv`)
  - GO back to step 1.

