#! /bin/bash
set -e

USER=paninski-lab
REPO_NAME=lightning-pose

BASE_DIR=/local/$(whoami)/builds
TARGET_DIR=$BASE_DIR/$(date '+%Y_%m_%d-%H_%M_%S')
CONDA_ENV=lp

# Check if the PR number argument is provided
if [ $# -eq 0 ]; then
  echo "Error: Pull request number is required as the first argument."
  echo "Usage: $0 <PR_NUMBER>"
  exit 1
fi

PR_NUMBER="$1"


ml Miniforge-24.7.1-2
conda activate $CONDA_ENV

# Remove builds older than 24 hours
find "$BASE_DIR" -maxdepth 1 -type d -mtime +0 -print0 | while IFS= read -r -d $'\0' directory; do
  # Skip the starting directory itself.
  if [[ "$directory" != "$BASE_DIR" ]]; then
      echo "Removing directory: $directory"
      rm -rf "$directory"
  fi
done

# Get the code of the PR.
# For efficiency, rather than cloning, it inits a blank repo
# and fetches just the code we need.
git init "$TARGET_DIR"
cd "$TARGET_DIR"
git remote add upstream "https://github.com/$USER/$REPO_NAME.git"
git fetch upstream "refs/pull/$PR_NUMBER/merge"
git checkout FETCH_HEAD

pytest
