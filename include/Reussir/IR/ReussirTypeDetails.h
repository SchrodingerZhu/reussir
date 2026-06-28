//===-- ReussirTypeDetails.h - Reussir type details impl --------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the types used in the Reussir dialect (internal
// details).
//
//===----------------------------------------------------------------------===//
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/LogicalResult.h>

#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirEnumAttrs.h"

namespace reussir {
namespace detail {

//===----------------------------------------------------------------------===//
// RecordTypeStorage
//===----------------------------------------------------------------------===//
//
// We manually define the storage class for RecordType to handle
// self-references in the members and memberIsField arrays. Named
// structures can be initialized as incomplete such that they can be referred
// by their own.
//
//===----------------------------------------------------------------------===//

struct RecordTypeStorage : public mlir::TypeStorage {
  llvm::ArrayRef<mlir::Type> members;
  llvm::ArrayRef<bool> memberIsField;
  mlir::StringAttr name;
  bool complete;
  reussir::RecordKind kind;
  reussir::Capability defaultCapability;

  using KeyTy = RecordTypeStorage;

  RecordTypeStorage(llvm::ArrayRef<mlir::Type> members,
                    llvm::ArrayRef<bool> memberIsField, mlir::StringAttr name,
                    bool complete, reussir::RecordKind kind,
                    reussir::Capability defaultCapability)
      : members(members), memberIsField(memberIsField), name(name),
        complete(complete), kind(kind), defaultCapability(defaultCapability) {}

  RecordTypeStorage(const KeyTy &key) = default;

  KeyTy getAsKey() const { return *this; }

  bool operator==(const KeyTy &other) const {
    if (name)
      return name == other.name && kind == other.kind;
    return members == other.members && memberIsField == other.memberIsField &&
           kind == other.kind && defaultCapability == other.defaultCapability &&
           complete == other.complete;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name, key.kind);
    return llvm::hash_combine(key.members, key.memberIsField, key.kind,
                              key.defaultCapability, key.complete);
  }

  static RecordTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    auto *storage =
        new (allocator.allocate<RecordTypeStorage>()) RecordTypeStorage(key);
    // The key's `members`/`memberIsField` point at caller-owned buffers (a
    // local `SmallVector` in the parser or the C API). Copy them into the
    // uniquer's allocator so the uniqued storage owns them; otherwise a record
    // built complete directly via `get` (rather than parsed-then-`mutate`d)
    // dangles and crashes on first use.
    storage->members = allocator.copyInto(key.members);
    storage->memberIsField = allocator.copyInto(key.memberIsField);
    return storage;
  }

  /// Mutates the members and attributes an identified record.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations. Anonymous records are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete record does not
  /// change the record.
  llvm::LogicalResult mutate(mlir::TypeStorageAllocator &allocator,
                             llvm::ArrayRef<mlir::Type> members,
                             llvm::ArrayRef<bool> memberIsField,
                             reussir::Capability defaultCapability) {

    // Anonymous records cannot mutate.
    if (!name)
      return llvm::failure();

    // Mutation of complete records are allowed if they change nothing.
    if (complete)
      return llvm::success(members == this->members &&
                           memberIsField == this->memberIsField &&
                           defaultCapability == this->defaultCapability);

    // Mutate incomplete records.
    this->members = allocator.copyInto(members);
    this->memberIsField = allocator.copyInto(memberIsField);
    this->defaultCapability = defaultCapability;
    this->complete = true;
    return llvm::success();
  }
};

} // namespace detail
} // namespace reussir
