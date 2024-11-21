#!/bin/bash

# Script to reproduce results

envs=(
	"walker2d-medium-v0"
	"walker2d-medium-expert-v0"
	"walker2d-medium-replay-v0"
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