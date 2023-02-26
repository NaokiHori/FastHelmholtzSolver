#if !defined(SOLVER_H)
#define SOLVER_H

#include <fftw3.h>
#include "sdecomp.h"

typedef struct {
  // pre-factor in front of "p" term of Helmholtz equation
  double cnst;
  // global array size
  int *glsizes;
  // buffers (pencils) and their sizes
  double *x1pncl_r;
  fftw_complex *x1pncl_c;
  fftw_complex *y1pncl_c;
  fftw_complex *z1pncl_c;
  int *x1pncl_r_mysizes;
  int *x1pncl_c_mysizes;
  int *y1pncl_c_mysizes;
  int *z1pncl_c_mysizes;
  int *z1pncl_c_offsets;
  // sdecomp transpose plans
  sdecomp_transpose_plan_t *trans_plan_x1_to_y1;
  sdecomp_transpose_plan_t *trans_plan_y1_to_z1;
  sdecomp_transpose_plan_t *trans_plan_z1_to_y1;
  sdecomp_transpose_plan_t *trans_plan_y1_to_x1;
  // fftw plans
  fftw_plan fftw_plan_x_f;
  fftw_plan fftw_plan_y_f;
  fftw_plan fftw_plan_z_f;
  fftw_plan fftw_plan_x_b;
  fftw_plan fftw_plan_y_b;
  fftw_plan fftw_plan_z_b;
} solver_t;

extern solver_t *solver_init(const size_t ndims, const size_t *glsizes, const sdecomp_info_t *info, const double cnst, double *x1pncl_r);
extern int solver_solve(solver_t *solver);
extern int solver_finalise(solver_t *solver);

#endif // SOLVER_H
