#!/usr/bin/env bash

# save --config-path or --config-dir or --config-name
config_arg=""
# save the rest of the hydra params abc=123 format
param_arg=""
# the proxy script that will run with ${config_args} and ${param_args}
touch test.$$.sh
chmod a+x test.$$.sh

# args must follow  follow --key value format
# $0 is the script name, $1 is the first arg, $2 is the second arg, etc.
# in our case $1 is --key, $2 is value etc.
# we have $I where I is the total number of args passed to the script. $# is the number of args.
while true; do
  # key
	key="${1}"
  # if key is empty, then break out of the loop
  if [ -z "${key}" ]; then break; fi
  value=${2}
  # parse script that will be run as the proxy
  if [ "${key}" == "--script" ]; then
    if [ ! -f ${value} ]; then
      echo "script ${value} is not found"
      exit 1
    fi
    cp ${value} test.$$.sh
  else 
    if [[ "${key}" != "--config-name" && "${key}" != "--config-dir" && "${key}" != "--config-path" ]]; then
      # remove the leading --
      key=${key:2}
      param_arg="${param_arg} ${key}=${value}"
    else
      # append to the config arg
      config_arg="${config_arg} ${key}=${value}"
    fi
  fi  
  # advance to the next set of args
  # assuming args are coming as --key value so jump 2 forward.
  shift
  shift
done  

# append text to the modified script
echo "${param_arg} ${config_arg}" >> test.$$.sh
# show on stdout for debugging
echo "Running"
cat ./test.$$.sh
# execute the modified script
./test.$$.sh | tee test.$$.out


