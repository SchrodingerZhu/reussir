//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Fuses a pattern match's consumption of its scrutinee into a *destructuring*
/// decrement (Koka's `dropn_reuse` shape).
///
/// The frontend lowers `match v { Ctor(a, b, ..) => .. }` into a borrowing
/// dispatch whose taken arm retains every bound member and then decrements the
/// scrutinee:
///
///   %ref = reussir.rc.borrow %v
///   reussir.record.dispatch(%ref) {
///     [k] -> ^arm(%payload):
///       %a = ref.load(ref.project %payload [i])
///       rc.inc %a                      // retain the binding
///       ...
///       rc.dec %v                      // release the box (transitive glue)
///   }
///
/// The retain/release pair over each bound member is pure count traffic on the
/// hot unique path: the box dies and its slots could simply transfer. This
/// pass erases the bound retains and tags the decrement with the arm's tag and
/// the bound member indices; `reussir-rc-decrement-expansion` then expands it
/// shallowly — unique: release only *unbound* rc members and take the box as a
/// token; shared: retain the bound members and drop the count. A preceding
/// increment cancels against the destructuring decrement in
/// `reussir-inc-dec-cancellation` by rematerializing the bound retains
/// (borrow semantics), which is what turns inlined read-only predicates like
/// rbtree's `is_red` into pure tag reads.
///
/// The same consumption shape appears without a pattern match: a
/// rebuild-style update of a compound box reads its fields, retains the ones
/// it keeps, and releases the box —
///
///   %front = ref.load(ref.project(rc.borrow %q, 1))
///   rc.inc %front                    // keeps the front list
///   %state = ref.load(ref.project(rc.borrow %q, 2))
///   ref.acquire (ref.spilled %state) // keeps the [value] state's children
///   ...
///   rc.dec %q
///   ... build the updated box from the loaded fields ...
///
/// (functional-queue's snoc/uncons over its Hood-Melville Queue). On the hot
/// unique path every one of those retains is answered by the release of the
/// same member inside the whole-record drop — pure count traffic, and for a
/// [value] member a whole tag-switch of child increments each way. The
/// second walk below fuses these into a *tagless* destructuring decrement
/// whose bound members index the compound record itself: unique — the kept
/// members transfer, only unkept managed members release; shared — the
/// retains rematerialize (see `rematerializeBoundRetains`).
///
//===----------------------------------------------------------------------===//

#include "Reussir/Analysis/AliasAnalysis.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Transformation/Passes.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRRCDISPATCHFUSIONPASS
#define GEN_PASS_DEF_REUSSIRPARTIALMOVEPASS
#include "Reussir/Transformation/Passes.h.inc"

namespace {

// The member index a retained value was extracted from, when `value` is a
// direct `ref.load(ref.project(payload, i))` of the arm's payload reference.
std::optional<int64_t> extractedMemberIndex(mlir::Value value,
                                            mlir::Value payloadRef) {
  auto load =
      llvm::dyn_cast_if_present<ReussirRefLoadOp>(value.getDefiningOp());
  if (!load)
    return std::nullopt;
  auto project = llvm::dyn_cast_if_present<ReussirRefProjectOp>(
      load.getRef().getDefiningOp());
  if (!project || project.getRef() != payloadRef)
    return std::nullopt;
  return project.getIndex().getSExtValue();
}

// Fuse one arm: find the scrutinee's release, collect the bound retains
// before it, and rewrite. Returns whether the arm was fused.
bool fuseArm(mlir::Region &region, int64_t tag,
             mlir::TypedValue<RcType> scrutinee) {
  if (!region.hasOneBlock() || region.front().getNumArguments() != 1)
    return false;
  mlir::Value payloadRef = region.front().getArgument(0);

  // Locate the first release of the scrutinee at the arm's top level. Every
  // op before it must leave the box's count alone: borrows, projections,
  // loads, and retains of *other* values are fine; anything else that
  // touches the scrutinee (or is opaque) aborts.
  ReussirRcDecOp dec;
  llvm::SmallVector<ReussirRcIncOp> boundIncs;
  for (mlir::Operation &op : region.front()) {
    if (auto candidate = llvm::dyn_cast<ReussirRcDecOp>(op)) {
      if (candidate.getRcPtr() == scrutinee) {
        dec = candidate;
        break;
      }
      // A release of a value loaded from the scrutinee's box would race the
      // shallow expansion; only accept releases of unrelated values.
      if (extractedMemberIndex(candidate.getRcPtr(), payloadRef))
        return false;
      continue;
    }
    if (auto inc = llvm::dyn_cast<ReussirRcIncOp>(op)) {
      if (extractedMemberIndex(inc.getRcPtr(), payloadRef))
        boundIncs.push_back(inc);
      continue;
    }
    if (llvm::is_contained(op.getOperands(), mlir::Value(scrutinee)) &&
        !llvm::isa<ReussirRcBorrowOp>(op))
      return false;
    // Region-bearing or opaque ops before the release: bail (the release
    // may be conditional or the box may escape).
    if (op.getNumRegions() > 0 || llvm::isa<mlir::CallOpInterface>(op))
      return false;
  }
  if (!dec || dec.isDestructuring())
    return false;
  if (dec.getNullableToken() && !dec.getNullableToken().use_empty())
    return false;

  llvm::SmallVector<int64_t> bound;
  for (ReussirRcIncOp inc : boundIncs)
    bound.push_back(*extractedMemberIndex(inc.getRcPtr(), payloadRef));

  mlir::OpBuilder builder(dec);
  dec.setDestructureTagAttr(builder.getIndexAttr(tag));
  dec.setBoundMembersAttr(builder.getDenseI64ArrayAttr(bound));
  for (ReussirRcIncOp inc : boundIncs)
    inc.erase();
  return true;
}

// The member index a value was extracted from, when it is a direct
// `ref.load(ref.project(rc.borrow(box), i))`.
std::optional<int64_t> loadedMemberIndex(mlir::Value value,
                                         mlir::TypedValue<RcType> box) {
  auto load =
      llvm::dyn_cast_if_present<ReussirRefLoadOp>(value.getDefiningOp());
  if (!load)
    return std::nullopt;
  auto project = llvm::dyn_cast_if_present<ReussirRefProjectOp>(
      load.getRef().getDefiningOp());
  if (!project)
    return std::nullopt;
  auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
      project.getRef().getDefiningOp());
  if (!borrow || borrow.getRcPtr() != box)
    return std::nullopt;
  return project.getIndex().getSExtValue();
}

// The member index a `ref.acquire` retains, when its reference is the
// member's slot itself or a spilled copy of the loaded member.
std::optional<int64_t> acquiredMemberIndex(ReussirRefAcquireOp acquire,
                                           mlir::TypedValue<RcType> box) {
  mlir::Value ref = acquire.getRef();
  if (auto spilled =
          llvm::dyn_cast_if_present<ReussirRefSpilledOp>(ref.getDefiningOp()))
    return loadedMemberIndex(spilled.getValue(), box);
  auto project =
      llvm::dyn_cast_if_present<ReussirRefProjectOp>(ref.getDefiningOp());
  if (!project)
    return std::nullopt;
  auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
      project.getRef().getDefiningOp());
  if (!borrow || borrow.getRcPtr() != box)
    return std::nullopt;
  return project.getIndex().getSExtValue();
}

struct CompoundExtraction {
  int64_t index;
  ReussirRefLoadOp load;
  mlir::Operation *retain = nullptr;
  llvm::SmallVector<mlir::Operation *> chain;
};

// Recognize one direct, owned projection from a compound box. The frontend's
// canonical borrow-extract shape retains an rc member immediately after its
// load, and retains a value member either through the projected slot or a
// spilled copy of the load. Keeping this matcher deliberately narrow makes
// every reference derived from the box movable as one closed chain.
std::optional<CompoundExtraction>
matchCompoundExtraction(ReussirRefProjectOp project,
                        ReussirRcDecOp terminalDec) {
  if (project->getBlock() != terminalDec->getBlock() ||
      !project->isBeforeInBlock(terminalDec))
    return std::nullopt;

  ReussirRefLoadOp load;
  ReussirRefAcquireOp directAcquire;
  for (mlir::Operation *user : project.getProjected().getUsers()) {
    if (auto candidate = llvm::dyn_cast<ReussirRefLoadOp>(user)) {
      if (load)
        return std::nullopt;
      load = candidate;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<ReussirRefAcquireOp>(user)) {
      if (directAcquire)
        return std::nullopt;
      directAcquire = candidate;
      continue;
    }
    return std::nullopt;
  }
  if (!load || load->getBlock() != terminalDec->getBlock() ||
      !load->isBeforeInBlock(terminalDec))
    return std::nullopt;

  CompoundExtraction extraction{
      project.getIndex().getSExtValue(), load, nullptr, {}};
  extraction.chain.push_back(project);
  extraction.chain.push_back(load);

  mlir::Type memberType = load.getValue().getType();
  if (isTriviallyCopyable(memberType)) {
    if (directAcquire)
      return std::nullopt;
    return extraction;
  }

  if (llvm::isa<RcType>(memberType)) {
    if (directAcquire)
      return std::nullopt;
    auto inc = llvm::dyn_cast_if_present<ReussirRcIncOp>(load->getNextNode());
    if (!inc || inc.getRcPtr() != load.getValue() ||
        !inc->isBeforeInBlock(terminalDec))
      return std::nullopt;
    extraction.retain = inc;
    extraction.chain.push_back(inc);
    return extraction;
  }

  if (directAcquire) {
    if (directAcquire->getBlock() != terminalDec->getBlock() ||
        !directAcquire->isBeforeInBlock(terminalDec))
      return std::nullopt;
    extraction.retain = directAcquire;
    extraction.chain.push_back(directAcquire);
    return extraction;
  }

  auto spilled =
      llvm::dyn_cast_if_present<ReussirRefSpilledOp>(load->getNextNode());
  if (!spilled || spilled.getValue() != load.getValue() ||
      !spilled.getSpilled().hasOneUse())
    return std::nullopt;
  auto acquire =
      llvm::dyn_cast_if_present<ReussirRefAcquireOp>(spilled->getNextNode());
  if (!acquire || acquire.getRef() != spilled.getSpilled() ||
      !acquire->isBeforeInBlock(terminalDec))
    return std::nullopt;
  extraction.retain = acquire;
  extraction.chain.push_back(spilled);
  extraction.chain.push_back(acquire);
  return extraction;
}

// Whether a value's type can transitively own the candidate root type. Calls
// consuming such a value may hide a uniqueness observation or a release of an
// alias. Recursive record graphs are cut by `seen`.
bool typeMayContain(mlir::Type type, RcType root,
                    llvm::DenseSet<mlir::Type> &seen) {
  if (type == root)
    return true;
  if (!seen.insert(type).second)
    return false;
  if (auto rc = llvm::dyn_cast<RcType>(type))
    return typeMayContain(rc.getElementType(), root, seen);
  if (auto ref = llvm::dyn_cast<RefType>(type))
    return typeMayContain(ref.getElementType(), root, seen);
  if (auto nullable = llvm::dyn_cast<NullableType>(type))
    return typeMayContain(nullable.getPtrTy(), root, seen);
  if (auto record = llvm::dyn_cast<RecordType>(type)) {
    if (!record.getComplete())
      return true;
    return llvm::any_of(record.getMembers(), [&](mlir::Type member) {
      return typeMayContain(member, root, seen);
    });
  }
  if (auto array = llvm::dyn_cast<ArrayType>(type))
    return typeMayContain(array.getElementType(), root, seen);
  if (auto cell = llvm::dyn_cast<CellType>(type))
    return typeMayContain(cell.getElementType(), root, seen);
  // Unknown nontrivial containers (notably closures) may hide the root in
  // their payload. Primitive and raw values cannot.
  return !isTriviallyCopyable(type);
}

bool typeMayContain(mlir::Type type, RcType root) {
  llvm::DenseSet<mlir::Type> seen;
  return typeMayContain(type, root, seen);
}

bool mayAliasRoot(mlir::Value value, mlir::TypedValue<RcType> root,
                  mlir::AliasAnalysis &aliasAnalysis) {
  // Distinct typed rc values cannot denote the same box. The in-tree alias
  // implementation intentionally falls back to MayAlias across RcType
  // parameters, so establish this cheap type fact before querying it.
  if (value.getType() != root.getType())
    return false;
  return aliasAnalysis.alias(value, root) != mlir::AliasResult::NoAlias;
}

// Advancing a release changes the count observed in the crossed interval.
// Reject every operation that could make that change semantically visible.
// Canonical extraction-chain operations are ignored by the caller.
bool isUnsafeAcrossAdvancedRelease(mlir::Operation *op,
                                   mlir::TypedValue<RcType> root,
                                   mlir::AliasAnalysis &aliasAnalysis) {
  if (llvm::isa<mlir::RegionBranchOpInterface>(op))
    return true;
  if (auto call = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
    return llvm::any_of(call->getOperands(), [&](mlir::Value operand) {
      return typeMayContain(operand.getType(), root.getType());
    });
  }
  if (auto view = llvm::dyn_cast<ReussirArrayWithUniqueViewOp>(op))
    return mayAliasRoot(view.getArray(), root, aliasAnalysis);
  if (auto uniqify = llvm::dyn_cast<ReussirClosureUniqifyOp>(op))
    return mayAliasRoot(uniqify.getClosure(), root, aliasAnalysis);
  if (auto apply = llvm::dyn_cast<ReussirClosureApplyOp>(op))
    return mayAliasRoot(apply.getClosure(), root, aliasAnalysis);
  if (auto eval = llvm::dyn_cast<ReussirClosureEvalOp>(op))
    return mayAliasRoot(eval.getClosure(), root, aliasAnalysis);
  if (auto unique = llvm::dyn_cast<ReussirRcIsUniqueOp>(op))
    return mayAliasRoot(unique.getRcPtr(), root, aliasAnalysis);
  if (auto inc = llvm::dyn_cast<ReussirRcIncOp>(op))
    return mayAliasRoot(inc.getRcPtr(), root, aliasAnalysis);
  if (auto dec = llvm::dyn_cast<ReussirRcDecOp>(op))
    return mayAliasRoot(dec.getRcPtr(), root, aliasAnalysis);
  if (auto fetch = llvm::dyn_cast<ReussirRcFetchOp>(op))
    return mayAliasRoot(fetch.getRcPtr(), root, aliasAnalysis);
  if (auto fetchSub = llvm::dyn_cast<ReussirRcFetchSubOp>(op))
    return mayAliasRoot(fetchSub.getRcPtr(), root, aliasAnalysis);
  if (auto set = llvm::dyn_cast<ReussirRcSetOp>(op))
    return mayAliasRoot(set.getRcPtr(), root, aliasAnalysis);
  if (auto drop = llvm::dyn_cast<ReussirRefDropOp>(op))
    return typeMayContain(drop.getRef().getType().getElementType(),
                          root.getType());
  if (auto set = llvm::dyn_cast<ReussirCellSetOp>(op))
    return typeMayContain(set.getCell().getType().getElementType(),
                          root.getType());
  if (llvm::isa<ReussirCellRmwOp, ReussirCellRdlockOp, ReussirRegionCleanupOp>(
          op))
    return true;
  return false;
}

// Schedule a projection-only compound owner at its true ownership cut. Every
// nontrivial member must transfer: otherwise advancing the dec across a call
// could run an observable member drop (including an FFI cleanup hook) early.
// The runtime count==1 split remains the dynamic guard that makes freeing the
// box itself safe when an unseen holder exists.
bool scheduleCompleteCompoundMove(ReussirRcDecOp dec,
                                  mlir::AliasAnalysis &aliasAnalysis) {
  if (dec.isDestructuring())
    return false;
  if (dec.getNullableToken() && !dec.getNullableToken().use_empty())
    return false;
  mlir::TypedValue<RcType> root = dec.getRcPtr();
  RcType type = root.getType();
  if (type.getCapability() != Capability::shared ||
      type.getAtomicKind() != AtomicKind::normal || type.isRegional())
    return false;
  auto record = llvm::dyn_cast<RecordType>(type.getElementType());
  if (!record || !record.isCompound() || !record.getComplete() ||
      !record.hasNoRegionalFields())
    return false;

  llvm::SmallVector<CompoundExtraction> extractions;
  llvm::SmallPtrSet<mlir::Operation *, 16> chainOps;
  for (mlir::Operation *user : root.getUsers()) {
    if (user == dec)
      continue;
    auto borrow = llvm::dyn_cast<ReussirRcBorrowOp>(user);
    if (!borrow || borrow->getBlock() != dec->getBlock() ||
        !borrow->isBeforeInBlock(dec))
      return false;
    chainOps.insert(borrow);
    for (mlir::Operation *borrowUser : borrow.getBorrowed().getUsers()) {
      auto project = llvm::dyn_cast<ReussirRefProjectOp>(borrowUser);
      if (!project)
        return false;
      auto extraction = matchCompoundExtraction(project, dec);
      if (!extraction)
        return false;
      for (mlir::Operation *chainOp : extraction->chain)
        chainOps.insert(chainOp);
      extractions.push_back(std::move(*extraction));
    }
  }
  if (extractions.empty())
    return false;

  mlir::Operation *cut = nullptr;
  for (CompoundExtraction &extraction : extractions) {
    for (mlir::Operation *user : extraction.load.getValue().getUsers()) {
      if (chainOps.contains(user))
        continue;
      if (user->getBlock() != dec->getBlock())
        return false;
      if (!cut || user->isBeforeInBlock(cut))
        cut = user;
    }
  }
  if (!cut || dec->isBeforeInBlock(cut))
    return false;

  llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> retainsByMember(
      record.getMembers().size());
  for (const CompoundExtraction &extraction : extractions) {
    if (extraction.index < 0 ||
        static_cast<size_t>(extraction.index) >= retainsByMember.size())
      return false;
    if (extraction.retain)
      retainsByMember[extraction.index].push_back(extraction.retain);
  }

  llvm::SmallVector<int64_t> boundMembers;
  llvm::SmallVector<mlir::Operation *> retainsToErase;
  for (auto [index, member, isField] :
       llvm::enumerate(record.getMembers(), record.getMemberIsField())) {
    mlir::Type projected = getProjectedType(
        member, isField, Capability::unspecified, type.getAtomicKind());
    if (isTriviallyCopyable(projected))
      continue;
    if (retainsByMember[index].empty())
      return false;
    boundMembers.push_back(index);
    // A unique box contributes exactly one owned reference per field. With
    // duplicate projections only one retain is the transferred owner; the
    // remaining retains are real copies and must survive.
    retainsToErase.push_back(retainsByMember[index].front());
  }
  if (boundMembers.empty())
    return false;

  // The advanced release is inserted immediately before `cut`, so inspect
  // every operation whose view of the root count could change. Extraction
  // chains are being rescheduled as part of the same ownership rewrite.
  for (mlir::Operation *cursor = cut; cursor && cursor != dec;
       cursor = cursor->getNextNode()) {
    if (chainOps.contains(cursor))
      continue;
    if (isUnsafeAcrossAdvancedRelease(cursor, root, aliasAnalysis))
      return false;
  }

  // Preserve the extraction chains' original relative order while moving
  // them together. Member counts remain exact: before the cut the root still
  // owns each field; after it the unique path transfers that owner and the
  // shared path rematerializes the erased retain.
  llvm::SmallVector<mlir::Operation *> orderedChain;
  for (mlir::Operation &op : *dec->getBlock())
    if (chainOps.contains(&op))
      orderedChain.push_back(&op);
  for (mlir::Operation *op : orderedChain)
    op->moveBefore(cut);
  dec->moveBefore(cut);

  mlir::OpBuilder builder(dec);
  dec.setBoundMembersAttr(builder.getDenseI64ArrayAttr(boundMembers));
  for (mlir::Operation *retain : retainsToErase)
    retain->erase();

  // Local, always-on postcondition for the PoC: no reference into the freed
  // box survives the new ownership cut, and every nontrivial member has one
  // transfer entry. Whole-function balance is additionally exercised by lit
  // and sanitizer coverage.
  for (mlir::Operation *user : root.getUsers()) {
    if (user == dec)
      continue;
    auto borrow = llvm::cast<ReussirRcBorrowOp>(user);
    if (!borrow->isBeforeInBlock(dec))
      llvm::report_fatal_error("partial move left a borrow after its root dec");
    for (mlir::Operation *borrowUser : borrow.getBorrowed().getUsers()) {
      auto project = llvm::cast<ReussirRefProjectOp>(borrowUser);
      if (!project->isBeforeInBlock(dec))
        llvm::report_fatal_error(
            "partial move left a projection after its root dec");
      for (mlir::Operation *projectUser : project.getProjected().getUsers())
        if (!projectUser->isBeforeInBlock(dec))
          llvm::report_fatal_error(
              "partial move left a projected reference live after its root "
              "dec");
    }
  }
  return true;
}

// Fuse a rebuild-style consumption of a compound box: scan backward from a
// plain `rc.dec` collecting member retains, stopping at the first op the
// retains cannot soundly move past (into the decrement's shared branch).
// Within the window only reads, spills, and unrelated retains may appear —
// nothing that could free the box, mutate its fields, or observe a count.
void fuseCompoundConsumption(ReussirRcDecOp dec) {
  if (dec.isDestructuring())
    return;
  RcType type = dec.getRcPtr().getType();
  if (type.getCapability() != Capability::shared ||
      type.getAtomicKind() != AtomicKind::normal || type.isRegional())
    return;
  auto recordType = llvm::dyn_cast<RecordType>(type.getElementType());
  if (!recordType || !recordType.isCompound() || !recordType.getComplete() ||
      !recordType.hasNoRegionalFields())
    return;

  llvm::SmallDenseMap<int64_t, llvm::SmallVector<mlir::Operation *, 2>, 4>
      retainsByMember;
  llvm::SmallVector<int64_t> bound;
  auto rememberRetain = [&](int64_t index, mlir::Operation *retain) {
    auto &memberRetains = retainsByMember[index];
    if (memberRetains.empty())
      bound.push_back(index);
    memberRetains.push_back(retain);
  };
  for (mlir::Operation *cursor = dec->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (auto inc = llvm::dyn_cast<ReussirRcIncOp>(cursor)) {
      // An increment of the box itself belongs to the plain inc/dec
      // cancellation; do not shadow it.
      if (inc.getRcPtr() == dec.getRcPtr())
        return;
      if (auto idx = loadedMemberIndex(inc.getRcPtr(), dec.getRcPtr()))
        rememberRetain(*idx, cursor);
      continue;
    }
    if (auto acquire = llvm::dyn_cast<ReussirRefAcquireOp>(cursor)) {
      if (auto idx = acquiredMemberIndex(acquire, dec.getRcPtr()))
        rememberRetain(*idx, cursor);
      continue;
    }
    // Reads of any box and spills of loaded values commute with the
    // retains; so does anything without memory effects.
    if (llvm::isa<ReussirRcBorrowOp, ReussirRefProjectOp, ReussirRefLoadOp,
                  ReussirRefSpilledOp, ReussirRecordCoerceOp,
                  ReussirRecordTagOp>(cursor))
      continue;
    if (cursor->getNumRegions() == 0 && mlir::isMemoryEffectFree(cursor))
      continue;
    break;
  }
  if (bound.empty())
    return;

  mlir::OpBuilder builder(dec);
  dec.setBoundMembersAttr(builder.getDenseI64ArrayAttr(bound));
  // The box owns one reference per field, so at most one retain per member
  // can be replaced by that transferred owner. Duplicate projections are
  // additional real copies and keep their remaining retains.
  for (int64_t index : bound)
    retainsByMember[index].front()->erase();
}

struct RcDispatchFusionPass
    : public impl::ReussirRcDispatchFusionPassBase<RcDispatchFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation()->walk([&](ReussirRecordDispatchOp dispatch) {
      auto borrow = llvm::dyn_cast_if_present<ReussirRcBorrowOp>(
          dispatch.getVariant().getDefiningOp());
      if (!borrow)
        return;
      mlir::TypedValue<RcType> scrutinee = borrow.getRcPtr();
      RcType rcType = scrutinee.getType();
      if (rcType.getAtomicKind() != AtomicKind::normal || rcType.isRegional())
        return;
      auto recordType = llvm::dyn_cast<RecordType>(rcType.getElementType());
      if (!recordType || !recordType.isVariant() || !recordType.getComplete())
        return;
      for (auto [tagSetAttr, region] :
           llvm::zip(dispatch.getTagSets(), dispatch.getRegions())) {
        auto tagSet = llvm::cast<mlir::DenseI64ArrayAttr>(tagSetAttr);
        if (tagSet.size() != 1)
          continue;
        // The arm's payload must be a plain record view: fused member
        // handling only understands value/shared members.
        auto payload =
            llvm::dyn_cast<RecordType>(recordType.getMembers()[tagSet[0]]);
        if (!payload || !payload.getComplete() ||
            !payload.hasNoRegionalFields())
          continue;
        fuseArm(region, tagSet[0], scrutinee);
      }
    });
  }
};

struct PartialMovePass
    : public impl::ReussirPartialMovePassBase<PartialMovePass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::AliasAnalysis aliasAnalysis(getOperation());
    registerAliasAnalysisImplementations(aliasAnalysis);
    llvm::SmallVector<ReussirRcDecOp> decrements;
    getOperation()->walk(
        [&](ReussirRcDecOp dec) { decrements.push_back(dec); });
    for (ReussirRcDecOp dec : decrements) {
      scheduleCompleteCompoundMove(dec, aliasAnalysis);
      fuseCompoundConsumption(dec);
    }
  }
};

} // namespace
} // namespace reussir
