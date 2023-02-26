#include <mpi.h>
#include "sdecomp.h"
#include "internal.h"


/**
 * @brief destruct a structure sdecomp_t
 * @param[in,out] info : structure to be cleaned up
 * @return             : error code
 */
int sdecomp_internal_destruct(sdecomp_info_t *info){
  MPI_Comm *comm = &(info->comm_cart);
  MPI_Comm_free(comm);
  sdecomp_internal_free(info);
  return 0;
}

