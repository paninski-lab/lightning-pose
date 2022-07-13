#!/usr/bin/env bash

# test locally
# scripts/grid-hpo.sh --script scripts/grid-run-test-1.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"
# scripts/grid-hpo.sh --script scripts/grid-run-test-2.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"
# scripts/grid-hpo.sh --script scripts/grid-run-hydra-1.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"

# real run
# grid run --localdir scripts/grid-hpo.sh --script scripts/grid-run-test-1.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"
# grid run --localdir scripts/grid-hpo.sh --script scripts/grid-run-test-2.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"
# grid run --localdir scripts/grid-run-hydra-1.sh --script scripts/grid-run-test.sh --training.rng_seed_data_pt "[1,2]" --dali.base.train.sequence_length "[4,5]"

arg=""
touch test.$$.sh
chmod a+x test.$$.sh

# args must follow  follow --key value format
while true; do
  # assume --key and remove the leading --
	key="${1:2}"
  if [ -z "${key}" ]; then break; fi
  value=$2
  # parse script 
  if [ "${key}" == "script" ]; then
    if [ ! -f ${value} ]; then
      echo "script ${value} is not found"
      exit 1
    fi
    cp ${value} test.$$.sh
  else
  # not script, append to the arg
    arg="${arg} ${key}=${value}"
  fi
  
  # advance to the next set of args
  shift
  shift
done  

# execute the modified script
echo "${arg}" >> test.$$.sh
./test.$$.sh



