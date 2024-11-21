#!/bin/bash

# Script to reproduce results

envs=(
	"halfcheetah-medium-v0"
	"halfcheetah-medium-replay-v0"    
	"halfcheetah-medium-expert-v0"
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