// Construct sdecomp_info_t
// Allocate and initialise (decompose domain) sdecomp_info_t,
//   and return a pointer to it

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <mpi.h>
#include "sdecomp.h"
#include "internal.h"


static int sanitise_ndims(const MPI_Comm comm_default, const size_t ndims){
  // check ndims, which should be 2 or 3
  if(2 != ndims && 3 != ndims){
    int mpirank = 0;
    MPI_Comm_rank(comm_default, &mpirank);
    if(0 == mpirank){
      FILE *fp = sdecomp_internal_fopen();
      if(NULL != fp){
        fprintf(fp,
            "ndims (%zu is given) should be one of:\n"
            "  2 (two  -dimensional domain)\n"
            "  3 (three-dimensional domain)\n",
            ndims
        );
      }
      sdecomp_internal_fclose(fp);
    }
    MPI_Barrier(comm_default);
    return 1;
  }
  return 0;
}

static bool check_auto_decomp(const size_t ndims, const size_t *dims){
  // when at least one of dims[dim] is non-zero,
  //   user specifies how the domain should be decomposed,
  //   i.e., "auto_decomp" is false
  for(size_t dim = 0; dim < ndims; dim++){
    if(0 != dims[dim]){
      return false;
    }
  }
  return true;
}

static int sanitise_dims(const MPI_Comm comm_default, const size_t ndims, const size_t *dims){
  // check first element of "dims", which should be 1
  if(1 != dims[0]){
    // user tries to decompose in x direction,
    //   which is not allowed
    int mpirank = 0;
    MPI_Comm_rank(comm_default, &mpirank);
    if(0 == mpirank){
      FILE *fp = sdecomp_internal_fopen();
      if(NULL != fp){
        fprintf(fp, "dims[0] (%zu is given) should be 1 since we do not decompose the domain in x direction\n", dims[0]);
        fprintf(fp, "Please initialise \"dims\" with all zeros if you want to distribute process automatically\n");
      }
      sdecomp_internal_fclose(fp);
    }
    MPI_Barrier(comm_default);
    return 1;
  }
  // check number of total processes
  int nprocs = 0;
  MPI_Comm_size(comm_default, &nprocs);
  int nprocs_user = 1;
  for(size_t dim = 0; dim < ndims; dim++){
    nprocs_user *= dims[dim];
  }
  if(nprocs != nprocs_user){
    int mpirank = 0;
    MPI_Comm_rank(comm_default, &mpirank);
    if(0 == mpirank){
      FILE *fp = sdecomp_internal_fopen();
      if(NULL != fp){
        fprintf(fp, "Number of processes in the default communicator: %d\n", nprocs);
        for(size_t dim = 0; dim < ndims; dim++){
          fprintf(fp, "dims[%zu](%zu) %c ", dim, dims[dim], dim == ndims - 1 ? 'x' : ':');
        }
        fprintf(fp, "%d\n", nprocs_user);
        fprintf(fp, "They should be identical\n");
      }
      sdecomp_internal_fclose(fp);
    }
    MPI_Barrier(comm_default);
    return 1;
  }
  return 0;
}

static MPI_Comm create_new_communicator(const MPI_Comm comm_default, const bool auto_decomp, const size_t ndims, const size_t *dims, const bool *periods){
  // number of total processes participating in this decomposition
  int nprocs = 0;
  MPI_Comm_size(comm_default, &nprocs);
  // number of processes in each dimension
  // NOTE: int (NOT size_t), since this is passed to MPI API
  int *dims_ = sdecomp_internal_calloc(ndims, sizeof(int));
  if(auto_decomp){
    for(size_t dim = 0; dim < ndims; dim++){
      // force 1st dimension NOT decomposed (assign 1)
      dims_[dim] = dim == 0 ? 1 : 0;
    }
    // let MPI library decompose domain
    MPI_Dims_create(nprocs, (int)ndims, dims_);
  }else{
    // use user-specified value
    for(size_t dim = 0; dim < ndims; dim++){
      dims_[dim] = (int)(dims[dim]);
    }
  }
  // periodicities in all dimensions
  // NOTE: int (NOT bool), since this is passed to MPI API
  int *periods_ = sdecomp_internal_calloc(ndims, sizeof(int));
  for(size_t dim = 0; dim < ndims; dim++){
    periods_[dim] = (int)(periods[dim]);
  }
  // MPI rank in comm_default is no longer important,
  //   so I allow to override it
  const int reorder = 1;
  // create communicator
  MPI_Comm comm_cart = MPI_COMM_NULL;
  MPI_Cart_create(comm_default, (int)ndims, dims_, periods_, reorder, &comm_cart);
  sdecomp_internal_free(dims_);
  sdecomp_internal_free(periods_);
  return comm_cart;
}

static int output_decomposition(const sdecomp_info_t *info){
  const MPI_Comm comm = info->comm_cart;
  const size_t ndims = info->ndims;
  int nprocs = 0;
  int myrank = 0;
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &myrank);
  if(0 == myrank){
    // output from main process whose rank is 0
    // check process distribution
    int *dims    = sdecomp_internal_calloc(ndims, sizeof(int));
    int *periods = sdecomp_internal_calloc(ndims, sizeof(int));
    int *dummy   = sdecomp_internal_calloc(ndims, sizeof(int));
    MPI_Cart_get(comm, (int)ndims, dims, periods, dummy);
    // check position of each process in Cartesian communicator
    int *coords = sdecomp_internal_calloc((size_t)nprocs * ndims, sizeof(int));
    for(int rank = 0; rank < nprocs; rank++){
      MPI_Cart_coords(comm, rank, (int)ndims, coords + (size_t)rank * ndims);
    }
    // output
    FILE *fp = sdecomp_internal_fopen();
    if(NULL != fp){
      fprintf(fp, "Number of processes: ");
      for(size_t dim = 0; dim < ndims; dim++){
        fprintf(fp, "%d%c", dims[dim], dim == ndims-1 ? '\n' : ' ');
      }
      fprintf(fp, "Periodicity: ");
      for(size_t dim = 0; dim < ndims; dim++){
        fprintf(fp, "%s%c", periods[dim] ? "true" : "false", dim == ndims-1 ? '\n' : ' ');
      }
      for(int rank = 0; rank < nprocs; rank++){
        fprintf(fp, "Rank %d is at (", rank);
        for(size_t dim = 0; dim < ndims; dim++){
          fprintf(fp, "%d%s", *(coords + (size_t)rank * ndims + dim), dim == ndims-1 ? ")\n" : ", ");
        }
      }
    }
    sdecomp_internal_fclose(fp);
    sdecomp_internal_free(dims);
    sdecomp_internal_free(periods);
    sdecomp_internal_free(dummy);
    sdecomp_internal_free(coords);
  }
  MPI_Barrier(comm);
  return 0;
}

/**
 * @brief construct a structure sdecomp_info_t
 * @param[in] comm_default : MPI communicator which contains all processes
 *                             participating in the decomposition
 *                             (normally MPI_COMM_WORLD)
 * @param[in] ndims        : number of dimensions of the target domain
 * @param[in] dims         : number of processes in each dimension
 * @param[in] periods      : periodicities in each dimension
 * @return                 : (success) a pointer to sdecomp_info_t
 *                           (failure) NULL pointer
 */
sdecomp_info_t *sdecomp_internal_construct(const MPI_Comm comm_default, const size_t ndims, const size_t *dims, const bool *periods){
  /* sanitise argument "ndims" */
  if(0 != sanitise_ndims(comm_default, ndims)){
    return NULL;
  }
  /* check decomposing domain automatically or user specifies it */
  bool auto_decomp = check_auto_decomp(ndims, dims);
  /* sanitise argument "dims" when user specifies how to decompose the domain */
  if(!auto_decomp){
    if(0 != sanitise_dims(comm_default, ndims, dims)){
      return NULL;
    }
  }
  /* create new communicator (x1 pencil) */
  MPI_Comm comm_cart = create_new_communicator(
      comm_default,
      auto_decomp,
      ndims,
      dims,
      periods
  );
  /* create sdecomp_info_t and assign members */
  sdecomp_info_t *info = sdecomp_internal_calloc(1, sizeof(sdecomp_info_t));
  info->ndims = ndims;
  info->comm_cart = comm_cart;
  /* write information for debug use */
  output_decomposition(info);
  return info;
}

