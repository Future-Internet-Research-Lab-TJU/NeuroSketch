#ifndef __AFS_UTIL_H
#define __AFS_UTIL_H

#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unordered_map>

static inline double now_us()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (tv.tv_sec * (uint64_t) 1000000000 + (double)tv.tv_nsec);
}

#endif