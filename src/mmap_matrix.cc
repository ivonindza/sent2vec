/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "mmap_matrix.h"

namespace fasttext {

MmapMatrix::MmapMatrix(const char* name, const int64_t m, const int64_t n, const size_t offset) {
  m_ = m;
  n_ = n;

  // Open the model file
  int fd = open(name, O_RDONLY);
  if (fd == -1) {
    perror("ERROR MmapMatrix::MmapMatrix: open failed");
    exit(-1);
  }

  // Offset must be page aligned
  size_t pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);
  map_size_ = m * n * sizeof(real) + (offset - pa_offset);

  // Mmap the matrix from the file
  map_ptr_ = mmap(NULL, map_size_, PROT_READ, MAP_PRIVATE, fd, pa_offset);
  if (map_ptr_ == (void*)-1) {
    perror("ERROR MmapMatrix::MmapMatrix: mmap failed");
    exit(-1);
  }

  data_ = (real*)((char*)map_ptr_ + (offset - pa_offset));

  // Close the file descriptor
  int ret = close(fd);
  if (ret == -1) {
    perror("ERROR MmapMatrix::MmapMatrix: close failed");
    exit(-1);
  }
}

MmapMatrix::~MmapMatrix() {
  // Unmap
  int ret = munmap(map_ptr_, map_size_);
  if (ret == -1) {
    perror("ERROR MmapMatrix::~MmapMatrix: munmap failed");
    exit(-1);
  }
  map_ptr_ = nullptr;
  data_ = nullptr;
}

}
