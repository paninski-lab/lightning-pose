Combine Grid HPO with Hydra Multirun

The process:
- Instead of using `grid run real_script_name.py`
- Proxy the real script using [scripts/grid-hpo.sh](scripts/grid-hpo.sh) 
- Prepare the proxy script that has real_script_name.py
- The proxy will execute `grid run scripts/grid-hpo.sh --script proxy.sh` 
- Grid HPO params will be added to the real script via the proxy.sh

# Review examples of proxy scripts

[grid-run-test-1.sh](scripts/grid-run-test-1.sh)

```
# test locally
scripts/grid-hpo.sh --script scripts/grid-run-test-1.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"

# run on grid
grid run --name run-hydra-test --dockerfile Dockerfile \
--localdir \
--datastore_name mirror-mouse -- \ 
scripts/grid-hpo.sh \
--script scripts/grid-run-test-1.sh \
--training.rng_seed_data_pt "[1,2]" \
--dali.base.train.sequence_length "[4,5]"
```

[grid-run-test-2.sh](scripts/grid-run-test-2.sh)

```
# test locally

scripts/grid-hpo.sh --script scripts/grid-run-test-2.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"

# run on grid

grid run --name run-hydra-test --dockerfile Dockerfile \
--localdir \
--datastore_name mirror-mouse -- \ 
scripts/grid-hpo.sh \
--script scripts/grid-run-test-2.sh \
--training.rng_seed_data_pt "[1,2]" \
--dali.base.train.sequence_length "[4,5]"
```

[grid-run-hydra-1.sh](scripts/grid-run-hydra-1.sh)

```
# test locally
scripts/grid-hpo.sh --script scripts/grid-run-hydra-1.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"

# run on grid

grid run --name run-hydra-test --dockerfile Dockerfile \
--localdir --instance_type p3.2xlarge \
--datastore_name mirror-mouse -- \ 
scripts/grid-hpo.sh \
--script scripts/grid-run-test-2.sh \
--training.rng_seed_data_pt "[1,2]" \
--dali.base.train.sequence_length "[4,5]"
```