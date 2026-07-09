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

// Approximate size-class (bin) map of the default reussir-rt allocator
// (mimalloc behind the natural-bin GlobalAlloc), as observed via
// mi_usable_size(mi_malloc(n)) for n in [1, 1024]:
//   <= 16      : 8-byte bins  {8, 16}
//   17 .. 128  : 16-byte bins {32, 48, ..., 128}
//   > 128      : each power-of-two octave splits into four bins
//                (129..256 step 32, 257..512 step 64, 513..1024 step 128, …)
// The exact geometry varies by mimalloc major (e.g. whether 24 gets its own
// bin), which is fine — see below.
//
// This is only a *hint*, not a correctness contract. TokenReuse consults it to
// choose which donor to prefer: a fixed-size donor sharing a bin with its
// acceptor scores above a cross-bin one, since reusing it avoids a moving
// realloc. It gates nothing in lowering — `token.realloc` always calls
// `__reussir_reallocate` (which copies if the allocator moves the block), so a
// stale or imprecise map only costs a suboptimal pairing (an avoidable copy),
// never a miscompile. Keep it roughly in step with the linked allocator to keep
// the heuristic sharp (see #325: the previous SnMalloc-derived model paired
// across mimalloc bins, so its "in-bin" reallocs copied anyway).
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
