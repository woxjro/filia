// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Michelson/MichelsonDialect.h"
#include "Michelson/MichelsonTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::michelson;

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Michelson/MichelsonOpsTypes.cpp.inc"

bool MichelsonType::classof(Type type) {
  return llvm::isa<MichelsonDialect>(type.getDialect());
}
