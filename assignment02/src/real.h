#pragma once
#ifdef USE_DOUBLE
  using real_t = double;
  #define REAL_CST(x) (x##l)
#else
  using real_t = float;
  #define REAL_CST(x) (x##f)
#endif