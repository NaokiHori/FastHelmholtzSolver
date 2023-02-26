#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include <fftw3.h>
#include "sdecomp.h"
#include "solver.h"
#include "save.h"


#if !defined M_PI
#define M_PI 3.1415926535897932384
#endif

int main(void){
  // spatial dimension
  const size_t ndims = 3;
  // launch MPI
  MPI_Init(NULL, NULL);
  // assign all processes to decompose domain
  size_t *dims    = calloc(ndims, sizeof(size_t));
  bool   *periods = calloc(ndims, sizeof(  bool));
  // auto decompose
  dims[0] = 0, dims[1] = 0, dims[2] = 0;
  // fully periodic domain
  periods[0] = true, periods[1] = true, periods[2] = true;
  sdecomp_info_t *info = sdecomp.construct(
      MPI_COMM_WORLD,
      ndims,
      dims,
      periods
  );
  free(dims);
  free(periods);
  // global domain sizes (physical domain size and number of grid points)
  const double l = 2. * M_PI;
  const size_t n = 128;
  double *lengths = calloc(ndims, sizeof(double));
  size_t *glsizes = calloc(ndims, sizeof(size_t));
  // resolutions (e.g. dx = lx / nx)
  double *deltas = calloc(ndims, sizeof(double));
  for(sdecomp_dir_t dim = 0; dim < ndims; dim++){
    lengths[dim] = l;
    glsizes[dim] = n;
    deltas[dim]  = l / n;
  }
  // get local domain sizes and offsets of my x1pencil
  size_t *mysizes = calloc(ndims, sizeof(size_t));
  size_t *offsets = calloc(ndims, sizeof(size_t));
  for(sdecomp_dir_t dim = 0; dim < ndims; dim++){
    const int mysize = sdecomp.get_pencil_mysize(
        info,
        SDECOMP_X1PENCIL,
        dim,
        glsizes[dim]
    );
    const int offset = sdecomp.get_pencil_offset(
        info,
        SDECOMP_X1PENCIL,
        dim,
        glsizes[dim]
    );
    mysizes[dim] = (size_t)mysize;
    offsets[dim] = (size_t)offset;
  }
  // allocate scalar field
  double *scalar = calloc(
      mysizes[0] * mysizes[1] * mysizes[2],
      sizeof(double)
  );
  double *answer = calloc(
      mysizes[0] * mysizes[1] * mysizes[2],
      sizeof(double)
  );
  // initialise Helmholtz solver
  const double cnst = 1.;
  solver_t *solver = solver_init(ndims, glsizes, info, cnst, scalar);
  // initialise scalar field
  for(size_t k = 0; k < mysizes[2]; k++){
    // convert local index to global index
    double z = (k + offsets[2]) * deltas[2];
    for(size_t j = 0; j < mysizes[1]; j++){
      // convert local index to global index
      double y = (j + offsets[1]) * deltas[1];
      for(size_t i = 0; i < mysizes[0]; i++){
        // convert local index to global index
        double x = (i + offsets[0]) * deltas[0];
        const size_t index =
          + k * mysizes[0] * mysizes[1]
          + j * mysizes[0]
          + i;
        // solve d^2 p / dx_i dx_i + cnst x p = q,
        // e.g.
        //   p = sin(x) * sin(y) * sin(z),
        //   q = -2 * p
        // NOTE: also change domain size when this part is modified
        answer[index] = + 1. * sin(x) * sin(y) * sin(z);
        scalar[index] = - 2. * answer[index];
      }
    }
  }
  // solve equation, scalar will be overwritten
  solver_solve(solver);
  // check error
  {
    const MPI_Comm comm_cart = sdecomp.get_comm_cart(info);
    double sum = 0.;
    for(size_t k = 0; k < mysizes[2]; k++){
      for(size_t j = 0; j < mysizes[1]; j++){
        for(size_t i = 0; i < mysizes[0]; i++){
          const size_t index =
            + k * mysizes[0] * mysizes[1]
            + j * mysizes[0]
            + i;
          sum += pow(scalar[index] - answer[index], 2.);
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
    int myrank = 0;
    MPI_Comm_rank(comm_cart, &myrank);
    if(0 == myrank){
      printf("residual: % .1e\n", sqrt(sum));
    }
  }
  solver_finalise(solver);
  // output field to files
  save(ndims, glsizes, lengths, info, scalar);
  // cleanup sdecomp library
  sdecomp.destruct(info);
  // close MPI
  MPI_Finalize();
  // cleanup
  free(glsizes);
  free(lengths);
  free(deltas);
  free(mysizes);
  free(offsets);
  free(scalar);
  free(answer);
  return 0;
}

