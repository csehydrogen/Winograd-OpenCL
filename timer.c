#include <stdio.h>
#include "timer.h"

static struct timespec t[8];

static int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
    if (x->tv_nsec < y->tv_nsec) {
        int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
        y->tv_nsec -= 1000000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_nsec - y->tv_nsec > 1000000000) {
        int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
        y->tv_nsec += 1000000000 * nsec;
        y->tv_sec -= nsec;
    }
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_nsec = x->tv_nsec - y->tv_nsec;
    return x->tv_sec < y->tv_sec;
}

void timer_start(int id) {
    clock_gettime(CLOCK_MONOTONIC, &t[id]);
}

double timer_end(int id, const char *s) {
    struct timespec x, y;
    clock_gettime(CLOCK_MONOTONIC, &x);
    timespec_subtract(&y, &x, &t[id]);
    double elapsed = y.tv_sec * 1e3 + y.tv_nsec / 1e6;
    printf("[%s] %f ms\n", s, elapsed);
    return elapsed;
}
