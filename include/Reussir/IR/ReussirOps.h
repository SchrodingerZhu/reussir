//===-- ReussirOps.h - Reussir dialect operations ---------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides the definitions for operations used in the Reussir
// dialect.
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef REUSSIR_IR_REUSSIROPS_H
#define REUSSIR_IR_REUSSIROPS_H

#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirTypes.h"

#define GET_OP_CLASSES
#include "Reussir/IR/ReussirOps.h.inc"

#endif // REUSSIR_IR_REUSSIROPS_H
