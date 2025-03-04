# About the flags:
#   -X is needed for github cancel to work.
#     Otherwise srun needs two successive ctrl-C / SIGINT to cancel
#   -u was found to enable colors. --pty would also work.
srun -X -u -t 1:00:00 -c4 --mem=8g --gres=gpu:2 buildbot/build.sh $@
