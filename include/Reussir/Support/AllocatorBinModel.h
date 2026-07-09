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

// Size-class (bin) map of the default reussir-rt allocator: mimalloc behind
// the natural-bin path, built with `MI_MAX_ALIGN_SIZE=8` (so it guarantees
// only 8-byte alignment and its small size-classes step by 8, not 16 — see
// crates/reussir-rt). Verified against mi_usable_size(mi_malloc(n)) for
// n in [1, 1024]:
//   <= 64      : 8-byte bins  {8, 16, 24, 32, 40, 48, 56, 64}
//   > 64       : each power-of-two octave splits into four bins
//                (65..128 step 16, 129..256 step 32, 257..512 step 64, …)
// The 8-granular small range is what lets a per-constructor 24-byte variant
// box occupy a real 24-byte block instead of rounding to 32 (the default
// `MI_MAX_ALIGN_SIZE=16` gives 16-granular bins {8,16,32,48,…} with no 24).
//
// This is only a *hint*, not a correctness contract: TokenReuse consults it to
// prefer a donor that shares a bin with its acceptor (so reuse avoids a moving
// realloc). It gates nothing in lowering — `token.realloc` always calls the
// runtime, which copies if the block moves — so a stale or imprecise map only
// costs a suboptimal pairing, never a miscompile. Keep it roughly in step with
// the linked allocator to keep the heuristic sharp (see #325: the previous
// SnMalloc-derived model paired across mimalloc bins, so its reallocs copied).
constexpr std::size_t allocatorBinSize(std::size_t size) {
  if (size <= 64)
    return (size + 7) & ~std::size_t{7};
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
