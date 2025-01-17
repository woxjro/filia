// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef MICHELSON_MICHELSONOPS_H
#define MICHELSON_MICHELSONOPS_H

#include "Michelson/MichelsonTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Michelson/MichelsonOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "Michelson/MichelsonOps.h.inc"


#endif // MICHELSON_MICHELSONOPS_H
