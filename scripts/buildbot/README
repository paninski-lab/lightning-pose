These scripts are used to run tests on axon.

.github/workflows/run-tests.yml calls

    ssh axon 'sh buildbot/build_srun.sh <PR NUMBER>'

from a self-hosted runner. You can run this locally too.

I. Setup of self-hosted runner:

    A self-hosted runner is required that has ssh access to axon.

    1. You need a server. I.e. AWS, a physical PC,
      but the axon cluster does not work for this as axon nodes do not support outbound internet. (to be verified!!!)

    2. The server should support `ssh axon` alias.
       I.e. ensure ~/.ssh/config contains:
       Host axon
           HostName axon-remote.rc.zi.columbia.edu
           User <YOUR UNI HERE>
           Port 55

    3. ssh must be setup for passwordless authentication via public/private keys.
      This is achieved by generating keys using ssh-keygen, and copying the public key over
      to your authorized key file in axon (ssh-copy-id).

    4. Install and run Github Self-hosted Runner software

       Follow the steps at Repository settings > Actions > Runners > New self hosted runner

II. Setup on axon (or other SLURM cluster):

    Setup the following directory structure using the files herein:
        ~/buildbot/build.sh
        ~/buildbot/build_srun.sh

    Ensure build.sh has executable permissions, as required by `srun`:
        chmod +x ~/buildbot/build.sh

    Setup your conda environment, and configure the variables in build.sh.

III. Test that this is working.

From a server that meets the requirements of the self-hosted runner in I, run:

    ssh axon 'sh buildbot/build_srun.sh <PR NUMBER>'

It should successfully run tests. If not, debug.

IV. A note on security and self-hosted runners.

    Ensure external contributors cannot just run whatever workflow they
    want on your self-hosted runner: they would be able to execute
    arbitrary code on axon with the full permissions of your UNI.

    The only way I've found to be able to control this is via the following setting in Github:

    Settings > Actions > General >
        Approval for running fork pull request workflows from contributors

    The default is "Require approval for first-time contributors", but for maximum security,
    consider further restricting to repo owners organization members.
