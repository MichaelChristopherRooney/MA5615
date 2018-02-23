#!/bin/bash

declare -a FILENAMES_IN=(
	"'data-gpu-float-size-1000.txt'"
	"'data-gpu-float-size-10000.txt'"
	"'data-gpu-copy-times.txt'"
)

declare -a FILENAMES_OUT=(
	"'graphs/gpu-float-size-1000-time-vs-block-size.eps'"
	"'graphs/gpu-float-size-10000-time-vs-block-size.eps'"
	"'graphs/gpu-data-copy-times.eps'"
)

declare -a Y_LIMS=(
	"11000"
	"350000"
	"220000"
)

declare -a X_STARTS=(
	"4"
	"4"
	"1000"
)

declare -a X_LIMS=(
	"1024"
	"1024"
	"10000"
)

for index in {0..2}
do
	f_in=${FILENAMES_IN[index]}
	f_out=${FILENAMES_OUT[index]}
	y_l=${Y_LIMS[index]}
	x_s=${X_STARTS[index]}
	x_l=${X_LIMS[index]}
	gnuplot -e "FILENAME_IN=$f_in; FILENAME_OUT=$f_out; Y_LIM=$y_l; X_START=$x_s; X_LIM=$x_l" graph.plg 
done
