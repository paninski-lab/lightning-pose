srun --unbuffered -t 1:00:00 -c4 --mem=8g --gres=gpu:1 buildbot/build.sh $@
