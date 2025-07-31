#pragma once
#ifndef REUSSIR_IR_REUSSIRATTRS_H
#define REUSSIR_IR_REUSSIRATTRS_H

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

#define GET_ATTRDEF_CLASSES
#include "Reussir/IR/ReussirAttrs.h.inc"

#endif // REUSSIR_IR_REUSSIRATTRS_H
