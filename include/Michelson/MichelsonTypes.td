// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES
#define MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES

include "Michelson/MichelsonDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Michelson_Type<string name, string typeMnemonic>
    : TypeDef<Michelson_Dialect, name, [], "::mlir::michelson::MichelsonType"> {
  let mnemonic = typeMnemonic;
}

def Michelson_Value : Michelson_Type<"Value", "value"> {
  let summary = "A value in the Michelson language.";
  let description = [{
    This denotes a Michelson variable value.  A value is immutable, but if it is
    a class, then it's fields can be changed.
  }];
}

def Michelson_Cell : Michelson_Type<"Cell", "cell"> {
  let summary = "A cell for storing a Michelson value.";
  let description = [{
    This is a mutable cell that stores a Michelson value or may be unassigned.
  }];
}

def Michelson_Scope : Michelson_Type<"Scope", "scope"> {
  let summary = "Michelson scope";
  let description = [{
    A scope represents a mapping from Michelson variable names to values.
  }];
}

def Michelson_ReturnValue : Michelson_Type<"ReturnValue", "return_value"> {
  let summary = "Michelson return value";
  let description = [{
    This is a Michelson value or an exception (which should derive from BaseException).
    It is used for returning from Michelson functions.
  }];
}

#endif // MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES
