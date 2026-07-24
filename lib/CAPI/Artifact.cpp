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
/// This file implements the Reussir artifact packaging C API: LTO bitcode
/// emission and archive writing.
///
//===----------------------------------------------------------------------===//

#include "Reussir-c/Artifact.h"

#include <llvm-c/Core.h>

#include <llvm/Analysis/ModuleSummaryAnalysis.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/Archive.h>
#include <llvm/Object/ArchiveWriter.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <string>
#include <vector>

namespace {
/// Copies `message` into a buffer the caller releases with
/// `LLVMDisposeMessage`, matching the LLVM-C error convention.
char *toCMessage(const llvm::Twine &message) {
  return LLVMCreateMessage(message.str().c_str());
}

/// Renders an `llvm::Error` (consuming it) into that same buffer.
char *toCMessage(llvm::Error error) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << error;
  llvm::consumeError(std::move(error));
  return toCMessage(text);
}
} // namespace

char *reussirWriteLtoBitcode(LLVMModuleRef module, const char *path,
                             ReussirLtoMode mode) {
  if (mode == ReussirLtoNone)
    return toCMessage("bitcode emission requires an LTO mode");

  llvm::Module &m = *llvm::unwrap(module);
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return toCMessage(llvm::Twine("cannot open ") + path + ": " + ec.message());

  if (mode == ReussirLtoThin) {
    // ThinLTO's unit of work is one module plus a summary of what it defines
    // and calls; the linker reads those summaries to decide what to import
    // where, so bitcode without one is invisible to the ThinLTO plugin.
    //
    // The full ThinLTOBitcodeWriterPass additionally splits modules carrying
    // type metadata (for link-time CFI and whole-program devirtualization).
    // Nothing reaches here with type metadata: the backend pipeline devirtualizes
    // speculatively and then lowers every `llvm.type.test` away before this
    // point (see Jit.cpp), precisely so no artifact — bitcode or native —
    // ever carries an intrinsic that instruction selection cannot handle.
    llvm::ProfileSummaryInfo psi(m);
    llvm::ModuleSummaryIndex index = llvm::buildModuleSummaryIndex(
        m, /*GetBFICallback=*/nullptr, /*PSI=*/&psi);
    // The hash keys ThinLTO's incremental cache; without it every link
    // recompiles every module.
    llvm::WriteBitcodeToFile(m, out, /*ShouldPreserveUseListOrder=*/false,
                             &index, /*GenerateHash=*/true);
  } else {
    llvm::WriteBitcodeToFile(m, out);
  }

  out.close();
  if (out.has_error())
    return toCMessage(llvm::Twine("cannot write ") + path + ": " +
                      out.error().message());
  return nullptr;
}

char *reussirWriteArchive(const char *path, const char *const *members,
                          size_t count, const char *triple) {
  std::vector<llvm::NewArchiveMember> archiveMembers;
  archiveMembers.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    // Deterministic members: zeroed mtime/uid/gid/perms, so the archive is a
    // pure function of its inputs.
    llvm::Expected<llvm::NewArchiveMember> member =
        llvm::NewArchiveMember::getFile(members[i], /*Deterministic=*/true);
    if (!member)
      return toCMessage(llvm::Twine("cannot add ") + members[i] +
                        " to the archive: " +
                        llvm::toString(member.takeError()));
    // Record members by base name, as `llvm-ar` does. The caller emits them
    // into a scratch directory, and a member name carrying that path would
    // both clutter linker diagnostics and make the archive differ between
    // runs that are otherwise identical. The StringRef points into the
    // member's own buffer identifier, which outlives the write.
    member->MemberName = llvm::sys::path::filename(member->MemberName);
    archiveMembers.push_back(std::move(*member));
  }

  // The archive flavor follows the *target*, not the host: a cross-compiled
  // static library must be readable by the target's linker.
  llvm::object::Archive::Kind kind =
      triple ? llvm::object::Archive::getDefaultKindForTriple(
                   llvm::Triple(llvm::StringRef(triple)))
             : llvm::object::Archive::getDefaultKind();

  if (llvm::Error error = llvm::writeArchive(
          path, archiveMembers, llvm::SymtabWritingMode::NormalSymtab, kind,
          /*Deterministic=*/true, /*Thin=*/false))
    return toCMessage(std::move(error));
  return nullptr;
}
