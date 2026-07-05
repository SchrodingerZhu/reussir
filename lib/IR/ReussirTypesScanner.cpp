//===-- ReussirTypesScanner.cpp - Scanner instructions for types -*- c++
//-*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements scanner instruction emission for Reussir types.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirTypes.h"

namespace reussir {
size_t
RecordType::emitScannerInstructions(llvm::SmallVectorImpl<int32_t> &buffer,
                                    const mlir::DataLayout &dataLayout,
                                    const scanner::EmitState &EmitState) const {
  using namespace scanner;
  if (isCompound()) {
    size_t scannedBytes = EmitState.scannedBytes;
    size_t cursorPosition = EmitState.cursorPosition;
    // Walk members in packed physical order: scan offsets must match the
    // converted struct layout.
    for (uint32_t logical : getPackedOrder(dataLayout)) {
      mlir::Type rawMember = getMembers()[logical];
      bool isField = getMemberIsField()[logical];
      if (!rawMember)
        continue;
      mlir::Type member = memberStorageType(getContext(), rawMember, isField);
      // advance to the next field
      llvm::TypeSize memberSize = dataLayout.getTypeSize(member);
      if (!memberSize.isFixed())
        llvm_unreachable("RecordType must have a fixed size");
      uint64_t memberSizeInBytes = memberSize.getFixedValue();
      uint64_t memberAlignment = dataLayout.getTypeABIAlignment(member);
      scannedBytes = llvm::alignTo(scannedBytes, memberAlignment);
      if (isField) {
        if (cursorPosition < scannedBytes) {
          buffer.push_back(advance(scannedBytes - cursorPosition));
          cursorPosition = scannedBytes;
        }
        buffer.push_back(field());
      } else {
        RecordType nestedRecordType = llvm::dyn_cast<RecordType>(rawMember);
        if (!isField && nestedRecordType &&
            nestedRecordType.getDefaultCapability() == Capability::value &&
            !nestedRecordType.hasNoRegionalFields()) {
          size_t newCursorPosition = nestedRecordType.emitScannerInstructions(
              buffer, dataLayout,
              {/*cursorPosition=*/cursorPosition,
               /*scannedBytes=*/scannedBytes});
          buffer.pop_back();
          cursorPosition = newCursorPosition;
        }
      }
      scannedBytes += memberSizeInBytes;
    }
    buffer.push_back(end());
    return cursorPosition;
  }

  // Variant record
  // For variant, we always emit a variant instruction at the start followed by
  // a skip table. All variants join at the final end instruction. There is no
  // early termination for variants.
  // The layout is as follows:
  // | variant | skip | skip | skip | ... | end |
  size_t scannedBytes = EmitState.scannedBytes;
  size_t cursorPosition = EmitState.cursorPosition;
  auto tagSize = dataLayout.getTypeSize(getTagType()).getFixedValue();
  // A fused-header variant keeps its tag past the 4-byte count slot; move
  // the cursor onto it before the variant read.
  size_t tagOffset = scannedBytes + (hasFusedHeader() ? 4 : 0);
  if (cursorPosition < tagOffset) {
    buffer.push_back(advance(tagOffset - cursorPosition));
    cursorPosition = tagOffset;
  }
  buffer.push_back(variant(tagSize));
  scannedBytes = tagOffset + tagSize;
  auto layoutInfo = getElementRegionLayoutInfo(dataLayout);
  scannedBytes = llvm::alignTo(scannedBytes, layoutInfo.alignment);
  size_t currentSkip = buffer.size();
  size_t currentVariantStart = currentSkip + getMembers().size();
  const size_t expectedEnd = scannedBytes + layoutInfo.size;
  // reserve space for skip instructions
  buffer.append(getMembers().size(), end());
  llvm::SmallVector<size_t> rewriteEndToSkip;
  for (auto [rawMember, isField] :
       llvm::zip(getMembers(), getMemberIsField())) {
    if (isField) {
      // this is a direct field
      if (cursorPosition < scannedBytes)
        buffer.push_back(advance(scannedBytes - cursorPosition));
      buffer.push_back(field());
      // Scanned bytes is the local cursor position
      if (scannedBytes < expectedEnd)
        buffer.push_back(advance(expectedEnd - scannedBytes));
      rewriteEndToSkip.push_back(buffer.size());
      buffer.push_back(end());
    } else {
      auto recordType = llvm::dyn_cast_or_null<RecordType>(rawMember);
      // that if the record is directly nested as a value and has mutable fields
      // to scan (TODO: this should not be possible but let's be conservative
      // here)
      if (!isField && recordType &&
          recordType.getDefaultCapability() == Capability::value &&
          !recordType.hasNoRegionalFields()) {
        // we need to scan nested record
        size_t newCursorPosition = recordType.emitScannerInstructions(
            buffer, dataLayout,
            {/*cursorPosition=*/cursorPosition,
             /*scannedBytes=*/scannedBytes});
        buffer.pop_back(); // remove the nested end
        if (newCursorPosition < expectedEnd)
          buffer.push_back(advance(expectedEnd - newCursorPosition));
        rewriteEndToSkip.push_back(buffer.size());
        buffer.push_back(end());
      } else {
        // there is nothing to scan, but we do need to skip to the end
        if (cursorPosition < expectedEnd)
          buffer.push_back(advance(expectedEnd - cursorPosition));
        rewriteEndToSkip.push_back(buffer.size());
        buffer.push_back(end());
      }
    }

    // update skips
    buffer[currentSkip] = skip(currentVariantStart - currentSkip);
    currentSkip++;
    currentVariantStart = buffer.size();
  }

  // Rewrite all end placeholder to skip to end
  rewriteEndToSkip.pop_back(); // last end is real end
  size_t skipTarget = buffer.size() - 1;
  for (size_t idx : rewriteEndToSkip)
    buffer[idx] = skip(skipTarget - idx);
  return expectedEnd;
}

} // namespace reussir
