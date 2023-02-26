#include <stdbool.h>
#include <mpi.h>
#include "sdecomp.h"
#include "internal.h"
#include "../internal.h"


/* persistent all-to-all-v */

struct sdecomp_alltoall_t_ {
  int nrequests;
  MPI_Request *requests;
};

sdecomp_alltoall_t *sdecomp_internal_alltoall_init(const int nprocs_2d, const char * restrict sendbuf, const int * restrict scounts, const int * restrict sdispls, char * restrict recvbuf, const int * restrict rcounts, const int * restrict rdispls, const sdecomp_uint_t size_of_element, MPI_Datatype mpi_datatype, MPI_Comm comm){
  // ref: ompi/mca/coll/base/coll_base_alltoallv.c
  const int tag = 0;
  // NOTE: verbose: 2 * (nprocs_2d - 1) and use memcpy for self comm.
  const int nrequests = 2 * nprocs_2d;
  MPI_Request * restrict requests = sdecomp_internal_calloc(nrequests, sizeof(MPI_Request));
  for(int rank = 0; rank < nprocs_2d; rank++){
    MPI_Recv_init(
        recvbuf + rdispls[rank] * size_of_element,
        rcounts[rank],
        mpi_datatype,
        rank,
        tag,
        comm,
        requests + 0 * nprocs_2d + rank
    );
  }
  for(int rank = 0; rank < nprocs_2d; rank++){
    MPI_Send_init(
        sendbuf + sdispls[rank] * size_of_element,
        scounts[rank],
        mpi_datatype,
        rank,
        tag,
        comm,
        requests + 1 * nprocs_2d + rank
    );
  }
  sdecomp_alltoall_t * restrict alltoall = sdecomp_internal_calloc(1, sizeof(sdecomp_alltoall_t));
  alltoall->nrequests = nrequests;
  alltoall->requests  = requests;
  return alltoall;
}

int sdecomp_internal_alltoall_exec(sdecomp_alltoall_t * restrict alltoall){
  const int nrequests   = alltoall->nrequests;
  MPI_Request * restrict requests = alltoall->requests;
  MPI_Startall(nrequests, requests);
  MPI_Waitall(nrequests, requests, MPI_STATUS_IGNORE);
  return 0;
}

int sdecomp_internal_alltoall_finalise(sdecomp_alltoall_t *alltoall){
  const int nrequests = alltoall->nrequests;
  MPI_Request *requests = alltoall->requests;
  for(int n = 0; n < nrequests; n++){
    MPI_Request_free(requests + n);
  }
  free(requests);
  sdecomp_internal_free(alltoall);
  return 0;
}

