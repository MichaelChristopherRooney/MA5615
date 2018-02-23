#!/bin/bash

# This script will get results for various matrix sizes.
# When getting CPU results the CUDA version will be disabled.
# When getting CUDA results the CPU version will be disabled.
# Note: you need to manually set up the type to float/double outside of this script
declare -a BLOCK_SIZES=("4" "8" "16" "32" "64" "128" "256" "512" "1024")
declare -a MAT_SIZES=("1000" "5000" "10000" "30000")
declare -a DATA_TYPES=("float" "double")

for t_index in {0..1}
do
	data_type=${DATA_TYPES[t_index]}
	make clean
	make all DATA_TYPE_USED=$data_type
	# First get CPU results for each size
	for m_index in {0..3}
	do
		size=${MAT_SIZES[m_index]}
		echo "Running CPU version with n=m=$size and data type = $data_type"
		./prog -n $size -m $size -t -g > results/cpu-size-$size-$data_type.txt
		echo "Done"
	done
	# Now for each block size get the CUDA result for each matrix size
	for b_index in  {0..8}
	do
		block_size=${BLOCK_SIZES[b_index]}
		for m_index in {0..2} #TODO use all sizes
		do
			size=${MAT_SIZES[m_index]}
			echo "Running CUDA version with n=m=$size and block size = $block_size and data type = $data_type"
			./prog -n $size -m $size -t -c -b $block_size > results/cuda-size-$size-block-size-$block_size-$data_type.txt
			echo "Done"
		done
	done
done

