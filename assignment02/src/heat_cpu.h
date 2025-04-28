#ifndef HEAT_CPU_H
#define HEAT_CPU_H

#include <vector>
#include "real.h"

void cpu_heat_propagation(std::vector<real_t>& A, std::vector<real_t>& B, int n, int m, int p);
void compute_cpu_row_averages(const std::vector<real_t>& mat, std::vector<real_t>& avg, int n, int m);

#endif // HEAT_CPU_H