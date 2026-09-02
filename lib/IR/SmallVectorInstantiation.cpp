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
/// llvm/ADT/SmallVector.h declares `extern template class SmallVectorBase<…>`
/// and expects LLVMSupport to provide the members. MSVC emits inline members
/// of an explicit instantiation definition lazily — only the ones its own
/// translation unit happens to use — so a cl-built LLVMSupport (the
/// conda-forge MSVC toolchain) lacks the rest (e.g. the constructor and
/// `set_size`), while clang honours the extern-template declaration and, in
/// unoptimized code, references them instead of inlining. Instantiate the
/// full member set here; the definitions are COMDAT, so the archive member
/// carrying MSVC's subset is simply never pulled in.
///
//===----------------------------------------------------------------------===//

#if defined(_WIN32) && defined(__clang__)

#include <llvm/ADT/SmallVector.h>

namespace llvm {
template class SmallVectorBase<uint32_t>;
#if SIZE_MAX > UINT32_MAX
template class SmallVectorBase<uint64_t>;
#endif
} // namespace llvm

#endif // defined(_WIN32) && defined(__clang__)
