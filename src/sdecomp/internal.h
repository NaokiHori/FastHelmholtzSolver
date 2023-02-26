#if !defined(SDECOMP_INTERNAL_H)
#define SDECOMP_INTERNAL_H

/*
 * FOR INTERNAL USE
 * DO NOT INCLUDE THIS HEADER
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "sdecomp.h"

struct sdecomp_info_t_ {
  MPI_Comm comm_cart;
  size_t ndims;
};

// kernel functions to decide local size / offset of pencils
extern int sdecomp_internal_kernel_get_mysize(const int num_total, const int nprocs, const int myrank);
extern int sdecomp_internal_kernel_get_offset(const int num_total, const int nprocs, const int myrank);

// memory management
extern void *sdecomp_internal_calloc(const size_t count, const size_t size);
extern void sdecomp_internal_free(void *ptr);
// logging
extern FILE *sdecomp_internal_fopen(void);
extern int sdecomp_internal_fclose(FILE *stream);

// constructor and destructor
extern sdecomp_info_t *sdecomp_internal_construct(const MPI_Comm comm_default, const size_t ndims, const size_t *dims, const bool *periods);
extern int sdecomp_internal_destruct(sdecomp_info_t *info);

// getters
extern int sdecomp_internal_get_nprocs(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir);
extern int sdecomp_internal_get_myrank(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir);
extern int sdecomp_internal_get_pencil_mysize(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir, const int nitems);
extern int sdecomp_internal_get_pencil_offset(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const sdecomp_dir_t dir, const int nitems);
extern MPI_Comm sdecomp_internal_get_comm_cart(const sdecomp_info_t *info);

// functions which take care of transpose operations
extern sdecomp_transpose_plan_t *sdecomp_internal_transpose_construct(const sdecomp_info_t *info, const bool is_forward, const sdecomp_pencil_t pencil, const size_t *glsizes, const size_t size_elem, const MPI_Datatype mpi_datatype);
extern int sdecomp_internal_transpose_execute(sdecomp_transpose_plan_t *plan, const void * restrict sendbuf, void * restrict recvbuf);
extern int sdecomp_internal_transpose_destruct(sdecomp_transpose_plan_t *plan);
extern int sdecomp_internal_transpose_test(sdecomp_transpose_plan_t *plan);

#endif // SDECOMP_INTERNAL_H
