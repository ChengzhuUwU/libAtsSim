#include "cpu_parallel.h"

#ifdef WIN32
int CPU_THREAD_NUM = 8;
#elif __APPLE__
int CPU_THREAD_NUM = 12;
#endif

// #ifdef WIN32
// int CPU_THREAD_NUM = (omp_get_num_procs() - 1) > 1 ? (omp_get_num_procs() - 1) : 1;
// int CPU_THREAD_NUM = (omp_get_num_procs() - 1) > 1 ? (omp_get_num_procs() - 1) : 1;
// int CPU_THREAD_NUM = (12 - 1) > 1 ? (12 - 1) : 1;
// #else
//     int CPU_THREAD_NUM = (omp_get_num_threads() - 1) > 1 ? (omp_get_num_threads() - 1) : 1;
// #endif
