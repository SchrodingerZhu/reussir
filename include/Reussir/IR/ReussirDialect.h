//===-- ReussirDialect.h - Reussir dialect definition ----------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file defines the Reussir dialect and its core functionality.
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef REUSSIR_IR_REUSSIRDIALECT_H
#define REUSSIR_IR_REUSSIRDIALECT_H

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dialect.h>

#include "Reussir/IR/ReussirDialect.h.inc"

#endif // REUSSIR_IR_REUSSIRDIALECT_H
