#include <stdlib.h>
#include "solver.h"
#include "sdecomp.h"


solver_t *solver_init(const size_t ndims, const size_t *glsizes, const sdecomp_info_t *info, const double cnst, double *x1pncl_r){
  solver_t *solver = calloc(1, sizeof(solver_t));
  solver->cnst = cnst;
  // buffers (pencils) and their sizes
  solver->x1pncl_r = x1pncl_r;
  fftw_complex **x1pncl_c = &solver->x1pncl_c;
  fftw_complex **y1pncl_c = &solver->y1pncl_c;
  fftw_complex **z1pncl_c = &solver->z1pncl_c;
  int **glsizesr         = &solver->glsizes;
  int **x1pncl_r_mysizes = &solver->x1pncl_r_mysizes;
  int **x1pncl_c_mysizes = &solver->x1pncl_c_mysizes;
  int **y1pncl_c_mysizes = &solver->y1pncl_c_mysizes;
  int **z1pncl_c_mysizes = &solver->z1pncl_c_mysizes;
  int **z1pncl_c_offsets = &solver->z1pncl_c_offsets;
  // sdecomp transpose plans
  sdecomp_transpose_plan_t **trans_plan_x1_to_y1 = &solver->trans_plan_x1_to_y1;
  sdecomp_transpose_plan_t **trans_plan_y1_to_z1 = &solver->trans_plan_y1_to_z1;
  sdecomp_transpose_plan_t **trans_plan_z1_to_y1 = &solver->trans_plan_z1_to_y1;
  sdecomp_transpose_plan_t **trans_plan_y1_to_x1 = &solver->trans_plan_y1_to_x1;
  // fftw plans
  fftw_plan *fftw_plan_x_f = &solver->fftw_plan_x_f;
  fftw_plan *fftw_plan_y_f = &solver->fftw_plan_y_f;
  fftw_plan *fftw_plan_z_f = &solver->fftw_plan_z_f;
  fftw_plan *fftw_plan_x_b = &solver->fftw_plan_x_b;
  fftw_plan *fftw_plan_y_b = &solver->fftw_plan_y_b;
  fftw_plan *fftw_plan_z_b = &solver->fftw_plan_z_b;
  const unsigned flag = FFTW_ESTIMATE;
  // first allocate buffers to store sizes of pencils
  *glsizesr         = calloc(ndims, sizeof(   int)); // in real space
  size_t *glsizesw  = calloc(ndims, sizeof(size_t)); // in wave space
  *x1pncl_r_mysizes = calloc(ndims, sizeof(   int));
  *x1pncl_c_mysizes = calloc(ndims, sizeof(   int));
  *y1pncl_c_mysizes = calloc(ndims, sizeof(   int));
  *z1pncl_c_mysizes = calloc(ndims, sizeof(   int));
  *z1pncl_c_offsets = calloc(ndims, sizeof(   int));
  // global sizes
  for(sdecomp_dir_t dim = 0; dim < ndims; dim++){
    (*glsizesr)[dim] = glsizes[dim];
  }
  glsizesw[0] = glsizes[0] / 2 + 1;
  glsizesw[1] = glsizes[1];
  glsizesw[2] = glsizes[2];
  // x1 pencils (real and complex)
  {
    const sdecomp_pencil_t pencil = SDECOMP_X1PENCIL;
    // real x1pencil
    (*x1pncl_r_mysizes)[0] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_XDIR, glsizes[0]        );
    (*x1pncl_r_mysizes)[1] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_YDIR, glsizes[1]        );
    (*x1pncl_r_mysizes)[2] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_ZDIR, glsizes[2]        );
    // complex x1pencil
    (*x1pncl_c_mysizes)[0] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_XDIR, glsizes[0] / 2 + 1);
    (*x1pncl_c_mysizes)[1] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_YDIR, glsizes[1]        );
    (*x1pncl_c_mysizes)[2] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_ZDIR, glsizes[2]        );
    *x1pncl_c = calloc(
        (*x1pncl_c_mysizes)[0] * (*x1pncl_c_mysizes)[1] * (*x1pncl_c_mysizes)[2],
        sizeof(fftw_complex)
    );
  }
  // y1 pencil (complex)
  {
    const sdecomp_pencil_t pencil = SDECOMP_Y1PENCIL;
    (*y1pncl_c_mysizes)[0] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_XDIR, glsizes[0] / 2 + 1);
    (*y1pncl_c_mysizes)[1] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_YDIR, glsizes[1]        );
    (*y1pncl_c_mysizes)[2] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_ZDIR, glsizes[2]        );
    *y1pncl_c = calloc(
        (*y1pncl_c_mysizes)[0] * (*y1pncl_c_mysizes)[1] * (*y1pncl_c_mysizes)[2],
        sizeof(fftw_complex)
    );
  }
  // z1 pencil (complex)
  {
    const sdecomp_pencil_t pencil = SDECOMP_Z1PENCIL;
    (*z1pncl_c_mysizes)[0] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_XDIR, glsizes[0] / 2 + 1);
    (*z1pncl_c_mysizes)[1] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_YDIR, glsizes[1]        );
    (*z1pncl_c_mysizes)[2] = sdecomp.get_pencil_mysize(info, pencil, SDECOMP_ZDIR, glsizes[2]        );
    (*z1pncl_c_offsets)[0] = sdecomp.get_pencil_offset(info, pencil, SDECOMP_XDIR, glsizes[0] / 2 + 1);
    (*z1pncl_c_offsets)[1] = sdecomp.get_pencil_offset(info, pencil, SDECOMP_YDIR, glsizes[1]        );
    (*z1pncl_c_offsets)[2] = sdecomp.get_pencil_offset(info, pencil, SDECOMP_ZDIR, glsizes[2]        );
    *z1pncl_c = calloc(
        (*z1pncl_c_mysizes)[0] * (*z1pncl_c_mysizes)[1] * (*z1pncl_c_mysizes)[2],
        sizeof(fftw_complex)
    );
  }
  // FFTW plans
  int length = 0;
  // forward and backward FFTs in x direction
  length = (int)glsizes[0];
  *fftw_plan_x_f = fftw_plan_many_dft_r2c(
      1, &length, (*x1pncl_r_mysizes)[1] * (*x1pncl_r_mysizes)[2],
       x1pncl_r, NULL, 1, length,
      *x1pncl_c, NULL, 1, length / 2 + 1,
      flag
  );
  *fftw_plan_x_b = fftw_plan_many_dft_c2r(
      1, &length, (*x1pncl_c_mysizes)[1] * (*x1pncl_c_mysizes)[2],
      *x1pncl_c, NULL, 1, length / 2 + 1,
       x1pncl_r, NULL, 1, length,
      flag
  );
  // forward and backward FFTs in y direction
  length = (int)glsizes[1];
  *fftw_plan_y_f = fftw_plan_many_dft(
      1, &length, (*y1pncl_c_mysizes)[2] * (*y1pncl_c_mysizes)[0],
      *y1pncl_c, NULL, 1, length,
      *y1pncl_c, NULL, 1, length,
      FFTW_FORWARD, flag
  );
  *fftw_plan_y_b = fftw_plan_many_dft(
      1, &length, (*y1pncl_c_mysizes)[2] * (*y1pncl_c_mysizes)[0],
      *y1pncl_c, NULL, 1, length,
      *y1pncl_c, NULL, 1, length,
      FFTW_BACKWARD, flag
  );
  // forward and backward FFTs in z direction
  length = (int)glsizes[2];
  *fftw_plan_z_f = fftw_plan_many_dft(
      1, &length, (*z1pncl_c_mysizes)[0] * (*z1pncl_c_mysizes)[1],
      *z1pncl_c, NULL, 1, length,
      *z1pncl_c, NULL, 1, length,
      FFTW_FORWARD, flag
  );
  *fftw_plan_z_b = fftw_plan_many_dft(
      1, &length, (*z1pncl_c_mysizes)[0] * (*z1pncl_c_mysizes)[1],
      *z1pncl_c, NULL, 1, length,
      *z1pncl_c, NULL, 1, length,
      FFTW_BACKWARD, flag
  );
  // pencil rotations
  // rotate pencil, from x1 to y1
  *trans_plan_x1_to_y1 = sdecomp.transpose.construct(
      info,
      true,
      SDECOMP_X1PENCIL,
      glsizesw,
      sizeof(fftw_complex),
      MPI_C_DOUBLE_COMPLEX
  );
  // rotate pencil, from y1 to z1
  *trans_plan_y1_to_z1 = sdecomp.transpose.construct(
      info,
      true,
      SDECOMP_Y1PENCIL,
      glsizesw,
      sizeof(fftw_complex),
      MPI_C_DOUBLE_COMPLEX
  );
  // rotate pencil, from z1 to y1
  *trans_plan_z1_to_y1 = sdecomp.transpose.construct(
      info,
      false,
      SDECOMP_Z1PENCIL,
      glsizesw,
      sizeof(fftw_complex),
      MPI_C_DOUBLE_COMPLEX
  );
  // rotate pencil, from y1 to x1
  *trans_plan_y1_to_x1 = sdecomp.transpose.construct(
      info,
      false,
      SDECOMP_Y1PENCIL,
      glsizesw,
      sizeof(fftw_complex),
      MPI_C_DOUBLE_COMPLEX
  );
  free(glsizesw);
  return solver;
}

