/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_MMAP_MATRIX_H
#define FASTTEXT_MMAP_MATRIX_H

#include <cstdint>
#include <istream>
#include <ostream>

#include "matrix.h"
#include "real.h"

namespace fasttext {

class MmapMatrix : public Matrix {
  private:
    void* map_ptr_;
    size_t map_size_;

  public:
    MmapMatrix(const char*, const int64_t, const int64_t, const size_t);
    ~MmapMatrix();

    // Disable copying, save and load
    MmapMatrix(const MmapMatrix&) = delete;
    MmapMatrix& operator=(const MmapMatrix&) = delete;
    void save(std::ostream&) = delete;
    void load(std::istream&) = delete;

    // Disable all methods that change data_
    void zero() = delete;
    void uniform(real) = delete;
    void addRow(const Vector&, int64_t, real) = delete;
    void multiplyRow(const Vector& nums, int64_t ib = 0, int64_t ie = -1) = delete;
    void divideRow(const Vector& denoms, int64_t ib = 0, int64_t ie = -1) = delete;
};

}

#endif
