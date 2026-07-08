//===-- AllocatorBinModel.h - default allocator size-class map -*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 With LLVM Exceptions OR MIT
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef REUSSIR_SUPPORT_ALLOCATORBINMODEL_H
#define REUSSIR_SUPPORT_ALLOCATORBINMODEL_H

#include <bit>
#include <cstddef>

namespace reussir {

// Size-class (bin) map of the default reussir-rt allocator: mimalloc v3
// behind the natural-bin GlobalAlloc. Verified against
// mi_usable_size(mi_malloc(n)) for n in [1, 1024]:
//   <= 16      : 8-byte bins  {8, 16}
//   17 .. 128  : 16-byte bins {32, 48, ..., 128}   (note: no 24-byte bin)
//   > 128      : each power-of-two octave splits into four bins
//                (129..256 step 32, 257..512 step 64, 513..1024 step 128, …)
//
// This is a contract with the *linked* allocator, not a heuristic: TokenReuse
// emits `token.realloc` pairings only for sizes that share a bin here, and
// the same-bin lowering reuses the block in place without consulting the
// runtime. If the rt default allocator (or its bin geometry) changes, this
// map must change with it — a mismatch silently turns "in-place" pairings
// into moving reallocs (see #325: the previous SnMalloc-derived model paired
// across mimalloc bins, each such realloc copying dead token bytes).
constexpr std::size_t allocatorBinSize(std::size_t size) {
  if (size <= 16)
    return (size + 7) & ~std::size_t{7};
  if (size <= 128)
    return (size + 15) & ~std::size_t{15};
  const std::size_t octave = std::bit_ceil(size);
  const std::size_t step = octave / 8;
  return (size + step - 1) & ~(step - 1);
}

// Conservative bound: beyond this the allocator serves objects from large /
// huge segments where the bin model (and in-place reuse) no longer applies.
inline constexpr std::size_t kAllocatorBinModelMax = 64 * 1024;

// Whether two (align, size) layouts are served from the same allocator bin,
// i.e. a block allocated for one satisfies the other in place. Restricted to
// the natural-alignment path (align <= 16): over-aligned allocations go
// through mi_malloc_aligned and give no bin guarantee.
constexpr bool sameAllocatorBin(std::size_t oldAlign, std::size_t oldSize,
                                std::size_t newAlign, std::size_t newSize) {
  if (oldAlign != newAlign || oldAlign > 16)
    return false;
  const auto alignedSize = [](std::size_t alignment, std::size_t size) {
    return ((alignment - 1) | (size - 1)) + 1;
  };
  const std::size_t oldAligned = alignedSize(oldAlign, oldSize);
  const std::size_t newAligned = alignedSize(newAlign, newSize);
  if (oldAligned > kAllocatorBinModelMax || newAligned > kAllocatorBinModelMax)
    return false;
  return allocatorBinSize(oldAligned) == allocatorBinSize(newAligned);
}

} // namespace reussir

#endif // REUSSIR_SUPPORT_ALLOCATORBINMODEL_H
