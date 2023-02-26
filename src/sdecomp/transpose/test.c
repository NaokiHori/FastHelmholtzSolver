#include <string.h>
#include <limits.h>
#include "sdecomp.h"
#include "../internal.h"
#include "internal.h"


/*
 * Take the following x1 pencil as an example:
 *
 *   +---------------+
 *   | 12 13   14 15 |
 *   | 08 09   10 11 |
 *   +---------------+
 *   | 04 05   06 07 |
 *   | 00 01   02 03 |
 *   +---------------+
 *
 * which are in memory of rank 0: 00 01 02 03 04 05 06 07
 *
 * y1 pencil after being transposed should be
 *
 *   +-------+-------+
 *   | 12 13 | 14 15 |
 *   | 08 09 | 10 11 |
 *   |       |       |
 *   | 04 05 | 06 07 |
 *   | 00 01 | 02 03 |
 *   +-------+-------+
 *
 * which are in memory of rank 0: 00 04 08 12 01 05 09 13
 *
 * This idea is generalised for arbitrary datatype and 3D arrays
 */

static unsigned char refer_to_answer(const size_t ndims, const size_t *glsizes, const size_t *offsets, const size_t *indices){
  size_t val = 0;
  if(2 == ndims){
    val =
      + (indices[1] + offsets[1]) * glsizes[0]
      + (indices[0] + offsets[0]);
  }else{
    val =
      + (indices[2] + offsets[2]) * glsizes[1] * glsizes[0]
      + (indices[1] + offsets[1]) * glsizes[0]
      + (indices[0] + offsets[0]);
  }
  return (unsigned char)(val % UCHAR_MAX);
}

static size_t *get_mysizes(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const size_t *glsizes){
  const size_t ndims = info->ndims;
  size_t *mysizes = sdecomp_internal_calloc(ndims, sizeof(size_t));
  for(size_t dim = 0; dim < ndims; dim++){
    const int mysize = sdecomp.get_pencil_mysize(info, pencil, (sdecomp_dir_t)dim, (int)glsizes[dim]);
    if(mysize < 0){
      return NULL;
    }
    mysizes[dim] = (size_t)mysize;
  }
  return mysizes;
}

static size_t *get_offsets(const sdecomp_info_t *info, const sdecomp_pencil_t pencil, const size_t *glsizes){
  const size_t ndims = info->ndims;
  size_t *offsets = sdecomp_internal_calloc(ndims, sizeof(size_t));
  for(size_t dim = 0; dim < ndims; dim++){
    const int offset = sdecomp.get_pencil_offset(info, pencil, (sdecomp_dir_t)dim, (int)glsizes[dim]);
    if(offset < 0){
      return NULL;
    }
    offsets[dim] = (size_t)offset;
  }
  return offsets;
}

static size_t get_nitems(const size_t ndims, const size_t *mysizes){
  size_t nitems = 1;
  for(size_t dim = 0; dim < ndims; dim++){
    nitems *= mysizes[dim];
  }
  return nitems;
}

static size_t *get_indices(const size_t ndims, const sdecomp_pencil_t pencil, const size_t *mysizes, size_t index){
  size_t *indices = sdecomp_internal_calloc(ndims, sizeof(size_t));
  if(2 == ndims){
    // x1 pencil
    //   const int index =
    //     + j * x1pencil_mysizes[0]
    //     + i;
    // y1 pencil
    //   const int index =
    //     + i * y1pencil_mysizes[0]
    //     + j;
    switch(pencil){
      case SDECOMP_X1PENCIL:
      case SDECOMP_X2PENCIL:
        indices[1] = index /  mysizes[0];
        indices[0] = index -  mysizes[0] * indices[1];
        return indices;
      case SDECOMP_Y1PENCIL:
      case SDECOMP_Y2PENCIL:
        indices[0] = index /  mysizes[1];
        indices[1] = index -  mysizes[1] * indices[0];
        return indices;
      case SDECOMP_Z1PENCIL:
      case SDECOMP_Z2PENCIL:
        return NULL;
    }
  }else{ // 3 == ndims
    // x1/2 pencils
    //   const int index =
    //     + k * x1pencil_mysizes[0] * x1pencil_mysizes[1]
    //     + j * x1pencil_mysizes[0]
    //     + i;
    // y1/2 pencils
    //   const int index =
    //     + i * y1pencil_mysizes[1] * y1pencil_mysizes[2]
    //     + k * y1pencil_mysizes[1]
    //     + j;
    // z1/2 pencils
    //   const int index =
    //     + j * z1pencil_mysizes[2] * z1pencil_mysizes[0]
    //     + i * z1pencil_mysizes[2]
    //     + k;
    switch(pencil){
      case SDECOMP_X1PENCIL:
      case SDECOMP_X2PENCIL:
        indices[2] = index / (mysizes[0] * mysizes[1]);
        index      = index - (mysizes[0] * mysizes[1]) * indices[2];
        indices[1] = index /  mysizes[0];
        indices[0] = index -  mysizes[0]               * indices[1];
        return indices;
      case SDECOMP_Y1PENCIL:
      case SDECOMP_Y2PENCIL:
        indices[0] = index / (mysizes[1] * mysizes[2]);
        index      = index - (mysizes[1] * mysizes[2]) * indices[0];
        indices[2] = index /  mysizes[1];
        indices[1] = index -  mysizes[1]               * indices[2];
        return indices;
      case SDECOMP_Z1PENCIL:
      case SDECOMP_Z2PENCIL:
        indices[1] = index / (mysizes[2] * mysizes[0]);
        index      = index - (mysizes[2] * mysizes[0]) * indices[1];
        indices[0] = index /  mysizes[2];
        indices[2] = index -  mysizes[2]               * indices[0];
        return indices;
    }
  }
  return NULL;

}

int sdecomp_internal_transpose_test(sdecomp_transpose_plan_t *plan){
  const sdecomp_info_t *info = plan->info;
  const size_t ndims = info->ndims;
  const size_t *glsizes = plan->glsizes;
  const sdecomp_uint_t size_of_element = plan->size_of_element;
  const sdecomp_pencil_t pencil_bef = plan->pencil_bef;
  const sdecomp_pencil_t pencil_aft = plan->pencil_aft;
  size_t *bef_mysizes = get_mysizes(info, pencil_bef, glsizes);
  size_t *bef_offsets = get_offsets(info, pencil_bef, glsizes);
  size_t *aft_mysizes = get_mysizes(info, pencil_aft, glsizes);
  size_t *aft_offsets = get_offsets(info, pencil_aft, glsizes);
  if(NULL == bef_mysizes) return 1;
  if(NULL == bef_offsets) return 1;
  if(NULL == aft_mysizes) return 1;
  if(NULL == aft_offsets) return 1;
  const size_t bef_nitems = get_nitems(ndims, bef_mysizes);
  const size_t aft_nitems = get_nitems(ndims, aft_mysizes);
  unsigned char *bef    = sdecomp_internal_calloc(bef_nitems, size_of_element);
  unsigned char *aft    = sdecomp_internal_calloc(aft_nitems, size_of_element);
  unsigned char *values = sdecomp_internal_calloc(         1, size_of_element);
  // assign values to x1 pencil
  for(size_t index = 0; index < bef_nitems; index++){
    size_t *indices = get_indices(ndims, pencil_bef, bef_mysizes, index);
    const unsigned char value = refer_to_answer(ndims, glsizes, bef_offsets, indices);
    sdecomp_internal_free(indices);
    for(sdecomp_uint_t n = 0; n < size_of_element; n++){
      values[n] = value;
    }
    memcpy(
        bef + index * size_of_element,
        values,
        size_of_element
    );
  }
  // transpose
  sdecomp.transpose.execute(plan, bef, aft);
  // check consistency
  int retval = 0;
  for(size_t index = 0; index < aft_nitems; index++){
    size_t *indices = get_indices(ndims, pencil_aft, aft_mysizes, index);
    const unsigned char value = refer_to_answer(ndims, glsizes, aft_offsets, indices);
    sdecomp_internal_free(indices);
    for(sdecomp_uint_t n = 0; n < size_of_element; n++){
      values[n] = value;
    }
    if(0 != memcmp(
      aft + index * size_of_element,
      values,
      size_of_element
    )){
      retval += 1;
    }
  }
  sdecomp_internal_free(values);
  sdecomp_internal_free(bef_mysizes);
  sdecomp_internal_free(aft_mysizes);
  sdecomp_internal_free(bef_offsets);
  sdecomp_internal_free(aft_offsets);
  sdecomp_internal_free(bef);
  sdecomp_internal_free(aft);
  return retval;
}

