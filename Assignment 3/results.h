#pragma once

struct cuda_results_s {
	float float_alloc_time;
	float double_alloc_time;
	float float_kernel_time;
	float double_kernel_time;
	float float_copy_time;
	float double_copy_time;
};
