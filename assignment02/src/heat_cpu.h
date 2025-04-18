#ifndef HEAT_CPU_H
#define HEAT_CPU_H

#include <vector>

void cpu_heat_propagation(std::vector<float>& A, std::vector<float>& B, int n, int m, int p);
void compute_cpu_row_averages(const std::vector<float>& mat, std::vector<float>& avg, int n, int m);

#endif // HEAT_CPU_H