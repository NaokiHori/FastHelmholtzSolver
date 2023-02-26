#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "sdecomp.h"
#include "solver.h"


static int min(const int a, const int b){
  return a < b ? a : b;
}

int solver_solve(solver_t *solver){
  fftw_execute(solver->fftw_plan_x_f);
  sdecomp.transpose.execute(
      solver->trans_plan_x1_to_y1,
      solver->x1pncl_c,
      solver->y1pncl_c
  );
  fftw_execute(solver->fftw_plan_y_f);
  sdecomp.transpose.execute(
      solver->trans_plan_y1_to_z1,
      solver->y1pncl_c,
      solver->z1pncl_c
  );
  fftw_execute(solver->fftw_plan_z_f);
  // solve equation in wave space
  const double cnst = solver->cnst;
  const int *z1pncl_c_mysizes = solver->z1pncl_c_mysizes;
  const int *z1pncl_c_offsets = solver->z1pncl_c_offsets;
  fftw_complex *z1pncl_c = solver->z1pncl_c;
  const int *glsizes = solver->glsizes;
  // FFT norm
  const double norm = 1.
    / glsizes[0]
    / glsizes[1]
    / glsizes[2];
  // now memory is z -> x -> y, from contiguous to sparse
  // for each y
  for(int j = 0; j < z1pncl_c_mysizes[1]; j++){
    const int jofs = z1pncl_c_offsets[1];
    // for each x
    for(int i = 0; i < z1pncl_c_mysizes[0]; i++){
      const int iofs = z1pncl_c_offsets[0];
      // for each z
      for(int k = 0; k < z1pncl_c_mysizes[2]; k++){
        const int kofs = z1pncl_c_offsets[2];
        const int index =
          + j * z1pncl_c_mysizes[2] * z1pncl_c_mysizes[0]
          + i * z1pncl_c_mysizes[2]
          + k;
        const int gi = min(i + iofs, glsizes[0] - i - iofs);
        const int gj = min(j + jofs, glsizes[1] - j - jofs);
        const int gk = min(k + kofs, glsizes[2] - k - kofs);
        const double w = cnst - gi * gi - gj * gj - gk * gk;
        if(fabs(w) < 1.){
          // force zero mean
          z1pncl_c[index] = 0.;
        }else{
          // normalise FFT
          z1pncl_c[index] *= 1. / w * norm;
        }
      }
    }
  }
  fftw_execute(solver->fftw_plan_z_b);
  sdecomp.transpose.execute(
      solver->trans_plan_z1_to_y1,
      solver->z1pncl_c,
      solver->y1pncl_c
  );
  fftw_execute(solver->fftw_plan_y_b);
  sdecomp.transpose.execute(
      solver->trans_plan_y1_to_x1,
      solver->y1pncl_c,
      solver->x1pncl_c
  );
  fftw_execute(solver->fftw_plan_x_b);
  return 0;
}

