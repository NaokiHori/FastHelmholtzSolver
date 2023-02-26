#include <stdlib.h>
#include "sdecomp.h"
#include "solver.h"


int solver_finalise(solver_t *solver){
  free(solver->glsizes);
  free(solver->x1pncl_r_mysizes);
  free(solver->x1pncl_c_mysizes);
  free(solver->y1pncl_c_mysizes);
  free(solver->z1pncl_c_mysizes);
  free(solver->z1pncl_c_offsets);
  free(solver->x1pncl_c);
  free(solver->y1pncl_c);
  free(solver->z1pncl_c);
  sdecomp.transpose.destruct(solver->trans_plan_x1_to_y1);
  sdecomp.transpose.destruct(solver->trans_plan_y1_to_z1);
  sdecomp.transpose.destruct(solver->trans_plan_z1_to_y1);
  sdecomp.transpose.destruct(solver->trans_plan_y1_to_x1);
  fftw_destroy_plan(solver->fftw_plan_x_f);
  fftw_destroy_plan(solver->fftw_plan_x_b);
  fftw_destroy_plan(solver->fftw_plan_y_f);
  fftw_destroy_plan(solver->fftw_plan_y_b);
  fftw_destroy_plan(solver->fftw_plan_z_f);
  fftw_destroy_plan(solver->fftw_plan_z_b);
  free(solver);
  return 0;
}

