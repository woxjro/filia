// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Michelson/MichelsonDialect.h"
#include "Michelson/MichelsonOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::michelson;

#include "Michelson/MichelsonOpsDialect.cpp.inc"

void MichelsonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Michelson/MichelsonOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Michelson/MichelsonOpsTypes.cpp.inc"
      >();
}
