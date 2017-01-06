#ifndef TIMER_H
#define TIMER_H

#include <time.h>

void timer_start(int id);
double timer_end(int id, const char *s);

#endif
