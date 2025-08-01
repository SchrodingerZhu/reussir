//===-- ReussirAttrs.h - Reussir dialect operations -------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides the definitions for the attributes used in the
// Reussir dialect.
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef REUSSIR_IR_REUSSIRATTRS_H
#define REUSSIR_IR_REUSSIRATTRS_H

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

#define GET_ATTRDEF_CLASSES
#include "Reussir/IR/ReussirAttrs.h.inc"

#endif // REUSSIR_IR_REUSSIRATTRS_H
