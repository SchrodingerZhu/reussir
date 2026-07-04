//===-- Blake3Symbol.h - BLAKE3-hashed symbol mangling ---------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Content-addressed symbol names for lowering-generated globals (panic
// messages, string prefixes, string dispatchers): a Rust-v0-style identifier
// `_RNvC<len(ns)><ns><len(digits)>[_]<digits>` whose digits are the base-62
// encoding of the payload's 256-bit BLAKE3 digest.
//
// The digest-to-number convention matches the Rust frontend's `StringToken`
// (`reussir-core`'s `utils::string` + `utils::b62`): each 8-byte chunk of the
// digest is combined little-endian into a word, word 0 is the most
// significant, and the alphabet is `0-9a-zA-Z`. Both sides therefore derive
// identical digits from the same payload, on any host.
//
//===----------------------------------------------------------------------===//
#ifndef REUSSIR_CONVERSION_BLAKE3SYMBOL_H
#define REUSSIR_CONVERSION_BLAKE3SYMBOL_H

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/BLAKE3.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

namespace reussir {

/// Folds a hasher's finalized 32-byte digest into four 64-bit words, word 0
/// most significant, bytes combined little-endian within each word.
inline std::array<uint64_t, 4> blake3Words(llvm::BLAKE3 &hasher) {
  std::array<uint8_t, LLVM_BLAKE3_OUT_LEN> digest = hasher.final();
  std::array<uint64_t, 4> words{};
  for (unsigned w = 0; w < 4; ++w)
    for (unsigned b = 0; b < 8; ++b)
      words[w] |= static_cast<uint64_t>(digest[w * 8 + b]) << (8 * b);
  return words;
}

/// The base-62 digits (`0-9a-zA-Z`, most significant first) of a 256-bit
/// number given word 0 most significant.
inline std::string base62Digits(std::array<uint64_t, 4> words) {
  static constexpr char alphabet[] =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  // APInt's array constructor takes the least-significant word first.
  llvm::APInt value(256, {words[3], words[2], words[1], words[0]});
  if (value.isZero())
    return std::string(1, alphabet[0]);
  std::string digits;
  while (!value.isZero()) {
    uint64_t rem;
    llvm::APInt::udivrem(value, 62, value, rem);
    digits.push_back(alphabet[rem]);
  }
  std::reverse(digits.begin(), digits.end());
  return digits;
}

/// The mangled symbol for already-hashed words. The `_` separator appears
/// only when the leading digit is numeric (a v0 identifier cannot start with
/// a digit), and the digit count excludes it — mirroring
/// `StringToken::mangle` on the Rust side.
inline std::string mangledBlake3Symbol(llvm::StringRef nameSpace,
                                       std::array<uint64_t, 4> words) {
  std::string digits = base62Digits(words);
  const char *sep = (digits[0] >= '0' && digits[0] <= '9') ? "_" : "";
  return "_RNvC" + std::to_string(nameSpace.size()) + nameSpace.str() +
         std::to_string(digits.size()) + sep + digits;
}

/// The mangled symbol for a single payload string.
inline std::string mangledBlake3Symbol(llvm::StringRef nameSpace,
                                       llvm::StringRef payload) {
  llvm::BLAKE3 hasher;
  hasher.update(payload);
  return mangledBlake3Symbol(nameSpace, blake3Words(hasher));
}

} // namespace reussir

#endif // REUSSIR_CONVERSION_BLAKE3SYMBOL_H
