#include <stdint.h>
#include <stdbool.h>
#include <mpi.h>
#include "sdecomp.h"
#include "snpyio.h"
#include "save.h"


static const char npy_u64[] = {"'<u8'"};
static const char npy_dbl[] = {"'<f8'"};
static MPI_Datatype dtype_dbl = MPI_DOUBLE;

static int save_domain(const size_t ndims, const size_t *glsizes, const double *lengths){
  // N.B. error handlings are omitted,
  //   and 8 == sizeof(size_t) is assumed
  const size_t shape[1] = {ndims};
  {
    FILE *fp = fopen("glsizes.npy", "w");
    snpyio_w_header(1, shape, npy_u64, false, fp);
    fwrite(glsizes, ndims, sizeof(size_t), fp);
    fclose(fp);
  }
  {
    FILE *fp = fopen("lengths.npy", "w");
    snpyio_w_header(1, shape, npy_dbl, false, fp);
    fwrite(lengths, ndims, sizeof(double), fp);
    fclose(fp);
  }
  return 0;
}

static int save_scalar(const size_t ndims, const size_t *glsizes, const sdecomp_info_t *info, const double *scalar){
  const int array_of_sizes   [] = {glsizes[2], glsizes[1], glsizes[0]};
  const int array_of_subsizes[] = {
    sdecomp.get_pencil_mysize(info, SDECOMP_X1PENCIL, SDECOMP_ZDIR, glsizes[2]),
    sdecomp.get_pencil_mysize(info, SDECOMP_X1PENCIL, SDECOMP_YDIR, glsizes[1]),
    sdecomp.get_pencil_mysize(info, SDECOMP_X1PENCIL, SDECOMP_XDIR, glsizes[0]),
  };
  const int array_of_starts  [] = {
    sdecomp.get_pencil_offset(info, SDECOMP_X1PENCIL, SDECOMP_ZDIR, glsizes[2]),
    sdecomp.get_pencil_offset(info, SDECOMP_X1PENCIL, SDECOMP_YDIR, glsizes[1]),
    sdecomp.get_pencil_offset(info, SDECOMP_X1PENCIL, SDECOMP_XDIR, glsizes[0]),
  };
  const char fname[] = {"scalar.npy"};
  const MPI_Comm comm_cart = sdecomp.get_comm_cart(info);
  int myrank = 0;
  MPI_Comm_rank(comm_cart, &myrank);
  // write header by main process
  size_t header_size = 0;
  if(0 == myrank){
    FILE *fp = fopen(fname, "w");
    header_size = snpyio_w_header(ndims, glsizes, npy_dbl, false, fp);
    fclose(fp);
  }
  MPI_Barrier(comm_cart);
  MPI_Bcast(&header_size, sizeof(size_t) / sizeof(uint8_t), MPI_BYTE, 0, comm_cart);
  if(0 == header_size) return 1;
  // open file in parallel
  MPI_File fh = NULL;
  const int retval = MPI_File_open(comm_cart, fname, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
  if(MPI_SUCCESS != retval) return 1;
  // create data type and set file view
  MPI_Datatype filetype = MPI_DATATYPE_NULL;
  MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, dtype_dbl, &filetype);
  MPI_Type_commit(&filetype);
  MPI_File_set_view(fh, (MPI_Offset)header_size, dtype_dbl, filetype, "native", MPI_INFO_NULL);
  // total number of local elements
  int count = 1;
  for(size_t n = 0; n < ndims; n++){
    count *= array_of_subsizes[n];
  }
  MPI_File_write_all(fh, scalar, count, dtype_dbl, MPI_STATUS_IGNORE);
  MPI_Type_free(&filetype);
  MPI_File_close(&fh);
  return 0;
}

int save(const size_t ndims, const size_t *glsizes, const double *lengths, const sdecomp_info_t *info, const double *scalar){
  // save field and relevant information to NPY files
  const MPI_Comm comm_cart = sdecomp.get_comm_cart(info);
  int myrank = 0;
  MPI_Comm_rank(comm_cart, &myrank);
  if(0 == myrank){
    save_domain(ndims, glsizes, lengths);
  }
  save_scalar(ndims, glsizes, info, scalar);
  return 0;
}

