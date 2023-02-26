#include "sdecomp.h"
#include "internal.h"


/* assign all "methods" (pointers to all internal functions) */
const sdecomp_t sdecomp = {
  .construct         = sdecomp_internal_construct,
  .destruct          = sdecomp_internal_destruct,
  .get_nprocs        = sdecomp_internal_get_nprocs,
  .get_myrank        = sdecomp_internal_get_myrank,
  .get_pencil_mysize = sdecomp_internal_get_pencil_mysize,
  .get_pencil_offset = sdecomp_internal_get_pencil_offset,
  .get_comm_cart     = sdecomp_internal_get_comm_cart,
  .transpose         = {
    .construct = sdecomp_internal_transpose_construct,
    .execute   = sdecomp_internal_transpose_execute,
    .destruct  = sdecomp_internal_transpose_destruct,
    .test      = sdecomp_internal_transpose_test,
  },
};

