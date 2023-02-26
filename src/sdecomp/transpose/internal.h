#if !defined(SDECOMP_INTERNAL_TRANSPOSE_H)
#define SDECOMP_INTERNAL_TRANSPOSE_H

#include <stdint.h> // uint32_t
#include <mpi.h>    // MPI_Request, MPI_Datatype, MPI_Comm

/* 32-bit integer used to describe heavily-accessed indices */

typedef uint32_t sdecomp_uint_t;

/* persistent all-to-all-v */

typedef struct  sdecomp_alltoall_t_ sdecomp_alltoall_t;

extern sdecomp_alltoall_t *sdecomp_internal_alltoall_init(const int nprocs_2d, const char * restrict sendbuf, const int * restrict scounts, const int * restrict sdispls, char * restrict recvbuf, const int * restrict rcounts, const int * restrict rdispls, const sdecomp_uint_t size_of_element, MPI_Datatype mpi_datatype, MPI_Comm comm);
extern int sdecomp_internal_alltoall_exec(sdecomp_alltoall_t * restrict alltoall);
extern int sdecomp_internal_alltoall_finalise(sdecomp_alltoall_t *alltoall);

struct sdecomp_transpose_plan_t_ {
  const sdecomp_info_t * restrict info;
  sdecomp_alltoall_t * restrict alltoall;
  // variables used by packing and unpacking
  bool is_forward;
  sdecomp_uint_t nprocs_2d;
  void * restrict sendbuf;
  void * restrict recvbuf;
  sdecomp_uint_t * restrict spnclsizes;
  sdecomp_uint_t * restrict rpnclsizes;
  sdecomp_uint_t * restrict scounts;
  sdecomp_uint_t * restrict sdispls;
  sdecomp_uint_t * restrict rcounts;
  sdecomp_uint_t * restrict rdispls;
  sdecomp_uint_t size_of_element;
  // variables used by tests
  sdecomp_pencil_t pencil_bef;
  sdecomp_pencil_t pencil_aft;
  size_t * restrict glsizes;
};

extern sdecomp_transpose_plan_t *sdecomp_internal_transpose_init_2d(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const size_t *glsizes, const sdecomp_uint_t size_of_element, const MPI_Datatype mpi_datatype);
extern sdecomp_transpose_plan_t *sdecomp_internal_transpose_init_3d(const sdecomp_info_t *info, const bool is_forward, const sdecomp_pencil_t pencil, const size_t *glsizes, const sdecomp_uint_t size_of_element, const MPI_Datatype mpi_datatype);

extern int sdecomp_internal_execute_2d(sdecomp_transpose_plan_t * restrict plan, const void * restrict sendbuf, void * restrict recvbuf);
extern int sdecomp_internal_execute_3d(sdecomp_transpose_plan_t * restrict plan, const void * restrict sendbuf, void * restrict recvbuf);

#endif // SDECOMP_INTERNAL_TRANSPOSE_H
