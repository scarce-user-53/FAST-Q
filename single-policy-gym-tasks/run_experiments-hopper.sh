#!/bin/bash

# Script to reproduce results

envs=(
	"hopper-medium-v0"
	"hopper-medium-expert-v0"
	"hopper-medium-replay-v0"
	)

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		python main.py \
		--env $env \
		--seed $i
	done
done