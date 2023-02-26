#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>
#include "sdecomp.h"
#include "../internal.h"
#include "internal.h"


/**
 * @brief initialise transpose plan
 * @param[in] info            : struct contains information of process distribution
 * @param[in] is_forward      : forward transpose (true) or backward (false)
 * @param[in] pencil          : type of pencil
 * @param[in] glsizes         : global array size in each dimension
 * @param[in] size_of_element : size of each element, e.g. sizeof(double)
 * @param[in] mpi_datatype    : corresponding MPI_Datatype of each element, e.g. MPI_DOUBLE
 * @return                    : (success) a pointer to the created plan (struct)
 *                              (failure) NULL pointer
 */
sdecomp_transpose_plan_t *sdecomp_internal_transpose_construct(const sdecomp_info_t *info, const bool is_forward, const sdecomp_pencil_t pencil, const size_t *glsizes, const size_t size_of_element, const MPI_Datatype mpi_datatype){
  const size_t ndims = info->ndims;
  // reject too big domain size
  for(size_t dim = 0; dim < ndims; dim++){
    if(INT_MAX <= glsizes[dim]){
      FILE *fp = sdecomp_internal_fopen();
      fprintf(fp, "ERROR: sdecomp.transpose.construct\n");
      fprintf(fp, "glsizes[%zu] = %zu should be smaller than INT_MAX: %d\n", dim, glsizes[dim], INT_MAX);
      sdecomp_internal_fclose(fp);
      return NULL;
    }
  }
  // reject too big element
  if(USHRT_MAX <= size_of_element){
    FILE *fp = sdecomp_internal_fopen();
    fprintf(fp, "ERROR: sdecomp.transpose.construct\n");
    fprintf(fp, "size_of_element = %zu should be smaller than USHRT_MAX: %d\n", size_of_element, USHRT_MAX);
    sdecomp_internal_fclose(fp);
    return NULL;
  }
  if(2 == ndims){
    // check pencil type for 2D
    if(
        SDECOMP_X1PENCIL != pencil &&
        SDECOMP_Y1PENCIL != pencil
    ){
      FILE *fp = sdecomp_internal_fopen();
      fprintf(fp, "ERROR: sdecomp.transpose.construct\n");
      fprintf(fp, "pencil is expected to be one of \n");
      fprintf(fp, "  SDECOMP_X1PENCIL (0)\n");
      fprintf(fp, "  SDECOMP_Y1PENCIL (1)\n");
      sdecomp_internal_fclose(fp);
      return NULL;
    }
    return sdecomp_internal_transpose_init_2d(info, pencil, glsizes, (sdecomp_uint_t)size_of_element, mpi_datatype);
  }else{
    // check pencil type for 3D
    if(
        SDECOMP_X1PENCIL != pencil &&
        SDECOMP_Y1PENCIL != pencil &&
        SDECOMP_Z1PENCIL != pencil &&
        SDECOMP_X2PENCIL != pencil &&
        SDECOMP_Y2PENCIL != pencil &&
        SDECOMP_Z2PENCIL != pencil
    ){
      FILE *fp = sdecomp_internal_fopen();
      fprintf(fp, "ERROR: sdecomp.transpose.construct\n");
      fprintf(fp, "pencil is expected to be one of \n");
      fprintf(fp, "  SDECOMP_X1PENCIL (0)\n");
      fprintf(fp, "  SDECOMP_Y1PENCIL (1)\n");
      fprintf(fp, "  SDECOMP_Z1PENCIL (2)\n");
      fprintf(fp, "  SDECOMP_X2PENCIL (3)\n");
      fprintf(fp, "  SDECOMP_Y2PENCIL (4)\n");
      fprintf(fp, "  SDECOMP_Z2PENCIL (5)\n");
      sdecomp_internal_fclose(fp);
      return NULL;
    }
    return sdecomp_internal_transpose_init_3d(info, is_forward, pencil, glsizes, (sdecomp_uint_t)size_of_element, mpi_datatype);
  }
}

/**
 * @brief execute transpose
 * @param[in]  plan    : transpose plan initialised by constructor
 * @param[in]  sendbuf : pointer to the input  buffer
 * @param[out] recvbuf : pointer to the output buffer
 * @return             : error code
 */
int sdecomp_internal_transpose_execute(sdecomp_transpose_plan_t * restrict plan, const void * restrict sendbuf, void * restrict recvbuf){
  const sdecomp_info_t * restrict info = plan->info;
  const size_t ndims = info->ndims;
  if(2 == ndims){
    sdecomp_internal_execute_2d(plan, sendbuf, recvbuf);
  }else{
    sdecomp_internal_execute_3d(plan, sendbuf, recvbuf);
  }
  return 0;
}

/**
 * @brief finalise transpose plan
 * @param[in,out] plan : transpose plan to be cleaned-up
 * @return             : error code
 */
int sdecomp_internal_transpose_destruct(sdecomp_transpose_plan_t *plan){
  sdecomp_internal_alltoall_finalise(plan->alltoall);
  sdecomp_internal_free(plan->scounts);
  sdecomp_internal_free(plan->rcounts);
  sdecomp_internal_free(plan->sdispls);
  sdecomp_internal_free(plan->rdispls);
  sdecomp_internal_free(plan->spnclsizes);
  sdecomp_internal_free(plan->rpnclsizes);
  sdecomp_internal_free(plan->sendbuf);
  sdecomp_internal_free(plan->recvbuf);
  sdecomp_internal_free(plan->glsizes);
  sdecomp_internal_free(plan);
  return 0;
}

