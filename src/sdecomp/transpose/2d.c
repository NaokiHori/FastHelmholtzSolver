#include <string.h>
#include <mpi.h>
#include "sdecomp.h"
#include "../internal.h"
#include "internal.h"


#define SDECOMP_INTERNAL_NDIMS 2

static int convert_glsizes_to_sizes(const sdecomp_pencil_t pencil, const size_t glsizes[SDECOMP_INTERNAL_NDIMS], int sizes[SDECOMP_INTERNAL_NDIMS]){
  // "glsizes" tell size of domain in each PHYSICAL direction (x, y)
  // convert them in terms of MEMORY order
  //                   <-- contiguous sparse -->
  // from x1 pencils : (glsizes[0],  glsizes[1])
  // from y1 pencils : (glsizes[1],  glsizes[0])
  switch(pencil){
    case SDECOMP_X1PENCIL:
      sizes[0] = (int)(glsizes[0]);
      sizes[1] = (int)(glsizes[1]);
      return 0;
    case SDECOMP_Y1PENCIL:
      sizes[0] = (int)(glsizes[1]);
      sizes[1] = (int)(glsizes[0]);
      return 0;
    case SDECOMP_Z1PENCIL:
    case SDECOMP_X2PENCIL:
    case SDECOMP_Y2PENCIL:
    case SDECOMP_Z2PENCIL:
      // should not reach here
      sizes[0] = 0;
      sizes[1] = 0;
      return 1;
  }
  // should not reach here
  return 1;
}

static sdecomp_pencil_t get_pencil_after_rotation(const sdecomp_pencil_t pencil_bef){
  return
      SDECOMP_X1PENCIL == pencil_bef
    ? SDECOMP_Y1PENCIL
    : SDECOMP_X1PENCIL;
}

/**
 * @brief initialise transpose plan for 2d
 * @param[in] info            : struct contains information of process distribution
 * @param[in] pencil_bef      : type of pencil to be (before) rotated,
 *                                SDECOMP_X1PENCIL or SDECOMP_Y1PENCIL
 * @param[in] glsizes         : global array size in each dimension
 * @param[in] size_of_element : size of each element, e.g., sizeof(double)
 * @param[in] mpi_datatype    : corresponding MPI_Datatype of each element, e.g., MPI_DOUBLE
 * @return                    : (success) a pointer to the created plan
 *                              (failure) NULL
 */
sdecomp_transpose_plan_t *sdecomp_internal_transpose_init_2d(const sdecomp_info_t *info, const sdecomp_pencil_t pencil_bef, const size_t *glsizes, const sdecomp_uint_t size_of_element, const MPI_Datatype mpi_datatype){
  // get 2d communicator,
  //   in which comm_2d collective comm. will be called
  // since now the domain is 2D, this is identical to
  //   the default Cartesian communicator comm_cart
  const MPI_Comm comm_2d = sdecomp.get_comm_cart(info);
  // number of total process and my position
  int nprocs_2d = 0;
  int myrank_2d = 0;
  {
#define SDECOMP_INTERNAL_LOCAL_NDIMS 2 // 2d communicator
    int    dims[SDECOMP_INTERNAL_LOCAL_NDIMS] = {0};
    int periods[SDECOMP_INTERNAL_LOCAL_NDIMS] = {0};
    int  coords[SDECOMP_INTERNAL_LOCAL_NDIMS] = {0};
    MPI_Cart_get(comm_2d, SDECOMP_INTERNAL_LOCAL_NDIMS, dims, periods, coords);
    // I am interested in the memory-sparse direction
    // since comm_2d is defined in PHYSICAL (not memory) order,
    //   1: physical y direction is what I should extract here
    nprocs_2d =   dims[1];
    myrank_2d = coords[1];
#undef SDECOMP_INTERNAL_LOCAL_NDIMS
  }
  // get number of grid points in all directions
  //   in memory order (NOT physical order x, y)
  // NOTE: should never fail as long as
  //   glsizes is sanitised at the entrypoint
  int sizes[SDECOMP_INTERNAL_NDIMS] = {0};
  convert_glsizes_to_sizes(pencil_bef, glsizes, sizes);
  // check I can safely decompose the domain into chunks,
  //   i.e. local chunk sizes are positive (0 is not accepted)
  // I do it here for early return (before allocating buffers)
  for(int rank = 0; rank < nprocs_2d; rank++){
    if(0 >= sdecomp_internal_kernel_get_mysize(sizes[0], nprocs_2d, rank)) return NULL;
    if(0 >= sdecomp_internal_kernel_get_mysize(sizes[1], nprocs_2d, rank)) return NULL;
  }
  // internal send/recv buffers to store intermediate data
  // allocate datasizes of pencils before/after rotated
  // NOTE: sizes are in memory order
  sdecomp_uint_t *spnclsizes = sdecomp_internal_calloc(SDECOMP_INTERNAL_NDIMS, sizeof(sdecomp_uint_t));
  sdecomp_uint_t *rpnclsizes = sdecomp_internal_calloc(SDECOMP_INTERNAL_NDIMS, sizeof(sdecomp_uint_t));
  spnclsizes[0] = (sdecomp_uint_t)sdecomp_internal_kernel_get_mysize(sizes[0],         1,         0);
  spnclsizes[1] = (sdecomp_uint_t)sdecomp_internal_kernel_get_mysize(sizes[1], nprocs_2d, myrank_2d);
  rpnclsizes[0] = (sdecomp_uint_t)sdecomp_internal_kernel_get_mysize(sizes[1],         1,         0);
  rpnclsizes[1] = (sdecomp_uint_t)sdecomp_internal_kernel_get_mysize(sizes[0], nprocs_2d, myrank_2d);
  void *sendbuf = sdecomp_internal_calloc(
      spnclsizes[0] * spnclsizes[1],
      size_of_element
  );
  void *recvbuf = sdecomp_internal_calloc(
      rpnclsizes[0] * rpnclsizes[1],
      size_of_element
  );
  // consider communication between "rank0"-th (my) pencil and "rank1"-th (other incl. my) pencils
  // compute numbers of items and displacements
  int *scounts_ = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(int));
  int *rcounts_ = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(int));
  int *sdispls_ = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(int));
  int *rdispls_ = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(int));
  sdecomp_uint_t *scounts = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(sdecomp_uint_t));
  sdecomp_uint_t *sdispls = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(sdecomp_uint_t));
  sdecomp_uint_t *rcounts = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(sdecomp_uint_t));
  sdecomp_uint_t *rdispls = sdecomp_internal_calloc((size_t)nprocs_2d, sizeof(sdecomp_uint_t));
  for(int yrrank_2d = 0; yrrank_2d < nprocs_2d; yrrank_2d++){
    // send
    {
      const int chunk_isize = sdecomp_internal_kernel_get_mysize(sizes[0], nprocs_2d, yrrank_2d);
      const int chunk_ioffs = sdecomp_internal_kernel_get_offset(sizes[0], nprocs_2d, yrrank_2d);
      const int chunk_jsize = sdecomp_internal_kernel_get_mysize(sizes[1], nprocs_2d, myrank_2d);
      scounts_[yrrank_2d] = chunk_isize * chunk_jsize;
      sdispls_[yrrank_2d] = chunk_ioffs * chunk_jsize;
      scounts [yrrank_2d] = (sdecomp_uint_t)(chunk_isize);
      sdispls [yrrank_2d] = (sdecomp_uint_t)(chunk_ioffs);
    }
    // recv
    {
      const int chunk_isize = sdecomp_internal_kernel_get_mysize(sizes[1], nprocs_2d, yrrank_2d);
      const int chunk_ioffs = sdecomp_internal_kernel_get_offset(sizes[1], nprocs_2d, yrrank_2d);
      const int chunk_jsize = sdecomp_internal_kernel_get_mysize(sizes[0], nprocs_2d, myrank_2d);
      rcounts_[yrrank_2d] = chunk_isize * chunk_jsize;
      rdispls_[yrrank_2d] = chunk_ioffs * chunk_jsize;
      rcounts [yrrank_2d] = (sdecomp_uint_t)(chunk_isize);
      rdispls [yrrank_2d] = (sdecomp_uint_t)(chunk_ioffs);
    }
  }
  // launch alltoall persistent communication
  sdecomp_alltoall_t *alltoall = sdecomp_internal_alltoall_init(
    nprocs_2d,
    sendbuf, scounts_, sdispls_,
    recvbuf, rcounts_, rdispls_,
    size_of_element, mpi_datatype, comm_2d
  );
  sdecomp_internal_free(scounts_);
  sdecomp_internal_free(rcounts_);
  sdecomp_internal_free(sdispls_);
  sdecomp_internal_free(rdispls_);
  const sdecomp_pencil_t pencil_aft = get_pencil_after_rotation(pencil_bef);
  // create plan
  sdecomp_transpose_plan_t *plan = sdecomp_internal_calloc(1, sizeof(sdecomp_transpose_plan_t));
  plan->alltoall = alltoall;
  // variables used by packing and unpacking
  plan->is_forward      = true;
  plan->nprocs_2d       = (sdecomp_uint_t)nprocs_2d;
  plan->spnclsizes      = spnclsizes;
  plan->rpnclsizes      = rpnclsizes;
  plan->sendbuf         = sendbuf;
  plan->recvbuf         = recvbuf;
  plan->scounts         = scounts;
  plan->rcounts         = rcounts;
  plan->sdispls         = sdispls;
  plan->rdispls         = rdispls;
  plan->size_of_element = size_of_element;
  // variables used by test
  plan->info       = info;
  plan->pencil_bef = pencil_bef;
  plan->pencil_aft = pencil_aft;
  plan->glsizes    = sdecomp_internal_calloc(SDECOMP_INTERNAL_NDIMS, sizeof(size_t));
  for(sdecomp_uint_t dim = 0; dim < SDECOMP_INTERNAL_NDIMS; dim++){
    plan->glsizes[dim] = glsizes[dim];
  }
  return plan;
}

static inline int pack(sdecomp_transpose_plan_t * restrict plan, const char * restrict sendbuf){
  const sdecomp_uint_t nprocs_2d = plan->nprocs_2d;
  const sdecomp_uint_t pnclisize = plan->spnclsizes[0];
  const sdecomp_uint_t pncljsize = plan->spnclsizes[1];
  const sdecomp_uint_t size_of_element = plan->size_of_element;
  const sdecomp_uint_t * restrict scounts = plan->scounts;
  const sdecomp_uint_t * restrict sdispls = plan->sdispls;
  char * restrict sendbuf_ = (char *)(plan->sendbuf);
  // loop for buffer
  //   1. chunk
  //   2.     y
  //   3.     x
  for(sdecomp_uint_t rank = 0; rank < nprocs_2d; rank++){
    const sdecomp_uint_t chnkisize = scounts[rank];
    const sdecomp_uint_t chnkioffs = sdispls[rank];
    for(sdecomp_uint_t j = 0; j < pncljsize; j++){
      for(sdecomp_uint_t i = 0; i < chnkisize; i++){
        const sdecomp_uint_t index_src = (
            + j * pnclisize
            + i + chnkioffs
        ) * size_of_element;
        const sdecomp_uint_t index_dst = (
            + pncljsize * chnkioffs
            + j * chnkisize
            + i
        ) * size_of_element;
        memcpy(
            sendbuf_ + index_dst,
            sendbuf  + index_src,
            size_of_element
        );
      }
    }
  }
  return 0;
}

static inline int unpack(sdecomp_transpose_plan_t * restrict plan, char * restrict recvbuf){
  const sdecomp_uint_t nprocs_2d = plan->nprocs_2d;
  const sdecomp_uint_t pnclisize = plan->rpnclsizes[0];
  const sdecomp_uint_t pncljsize = plan->rpnclsizes[1];
  const sdecomp_uint_t size_of_element = plan->size_of_element;
  const sdecomp_uint_t * restrict rcounts = plan->rcounts;
  const sdecomp_uint_t * restrict rdispls = plan->rdispls;
  char * restrict recvbuf_ = (char *)(plan->recvbuf);
  // loop for buffer
  //   1. chunk
  //   2.     x
  //   3.     y
  for(sdecomp_uint_t rank = 0; rank < nprocs_2d; rank++){
    const sdecomp_uint_t chnkisize = rcounts[rank];
    const sdecomp_uint_t chnkioffs = rdispls[rank];
    for(sdecomp_uint_t i = 0; i < chnkisize; i++){
      for(sdecomp_uint_t j = 0; j < pncljsize; j++){
        const sdecomp_uint_t index_dst = (
            + j * pnclisize
            + i + chnkioffs
        ) * size_of_element;
        const sdecomp_uint_t index_src = (
            + chnkioffs * pncljsize
            + i * pncljsize
            + j
        ) * size_of_element;
        memcpy(
            recvbuf  + index_dst,
            recvbuf_ + index_src,
            size_of_element
        );
      }
    }
  }
  return 0;
}

int sdecomp_internal_execute_2d(sdecomp_transpose_plan_t * restrict plan, const void * restrict sendbuf, void * restrict recvbuf){
  pack(plan, sendbuf);
  sdecomp_internal_alltoall_exec(plan->alltoall);
  unpack(plan, recvbuf);
  return 0;
}

#undef SDECOMP_INTERNAL_NDIMS
