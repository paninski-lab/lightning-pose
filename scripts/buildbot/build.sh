#! /bin/bash
# Setup a SIGINT handler. Not sure why, but this is necessary for SIGINT (Ctrl-C) to cancel this script.
handle_sigint() {
    echo "Caught SIGINT (Ctrl+C). Exiting..."
    exit 130  # Exit with a specific code (128 + signal number)
}
# Trap the SIGINT signal and call the handle_sigint function
trap handle_sigint SIGINT

set -e

USER=paninski-lab
REPO_NAME=lightning-pose

BASE_DIR=/local/$(whoami)/builds
TARGET_DIR=$BASE_DIR/$(date '+%Y_%m_%d-%H_%M_%S')
CONDA_ENV=lp_build

PR_NUMBER="${1:-0}"

echo "Running from $(hostname)"

# Activate environment
echo "Setting up environment..."
source ~/.bashrc
ml Miniforge-24.7.1-2
conda activate $CONDA_ENV
ml gcc/14.1                        # satisfies NumPy >= 2 requirement
export LD_PRELOAD=/home/$(whoami)/.conda/envs/$CONDA_ENV/lib/libstdc++.so.6
echo "Active conda environment: $CONDA_ENV"
echo "Python location: $(which python)"
echo "Pip location $(which pip)"

# Remove builds older than 24 hours
find "$BASE_DIR" -maxdepth 1 -type d -mtime +0 -print0 | while IFS= read -r -d $'\0' directory; do
  # Skip the starting directory itself.
  if [[ "$directory" != "$BASE_DIR" ]]; then
      echo "Removing directory: $directory"
      rm -rf "$directory"
  fi
done

# Get the code. For efficiency, init a blank repo and fetch only what we need.
git init "$TARGET_DIR"
cd "$TARGET_DIR"
git remote add upstream "https://github.com/$USER/$REPO_NAME.git"
if [ "$PR_NUMBER" -eq 0 ]; then
  echo "No PR number provided; checking out main."
  git fetch upstream main
  git checkout FETCH_HEAD
else
  git fetch upstream "refs/pull/$PR_NUMBER/merge"
  git checkout FETCH_HEAD
fi

# Install with checks
pip install ".[dev]" # Install any new dependencies.
echo "Pip install exit code: $?"
pip show lightning_pose
python -c "import lightning_pose; print('LP location:', lightning_pose.__file__); print('LP import successful')"

# Run with html reporting.
pytest --html=report.html --self-contained-html --cov=. --cov-report=xml:$HOME/buildbot_lp/coverage.xml tests/
