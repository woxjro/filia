// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES
#define MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES

include "Michelson/MichelsonDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Michelson_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Michelson_Dialect, name, [], "::mlir::michelson::MichelsonType"> {
  let mnemonic = typeMnemonic;
}

def Michelson_Mutez: Michelson_Type<"Mutez", "mutez"> {
    let summary = "Michelson mutez type";
    let description = [{
        The `!michelson.mutez` type is an Michelson mutez type.

        Example:
        ```mlir
        !michelson.mutez
        ```
    }];
    // let parameters = (ins);
}

def Michelson_Unit: Michelson_Type<"Unit", "unit"> {
    let summary = "Michelson unit type";
    let description = [{
        The `!michelson.unit` type is an Michelson unit type.

        Example:
        ```mlir
        !michelson.unit
        ```
    }];
}


def Michelson_Operation : Michelson_Type<"Operation", "operation"> {
  let summary = "Michelson operation type";
  let description = [{
    The `!michelson.operation` type is an Michelson operation type.

    Example:
    ```mlir
    !michelson.operation
    ```
  }];
}

def Michelson_StructType :
    DialectType<Michelson_Dialect, CPred<"$_self.isa<StructType>()">,
                "Michelson struct type">;

def Michelson_PrimitiveType : Type<
  CPred<"::llvm::isa<::mlir::michelson::MichelsonType>($_self)">, "michelson type",
        "::mlir::michelson::MichelsonType">;


def Michelson_List: Michelson_Type<"List", "list"> {
  let summary = "Michelson List with a parameterized element type";
  let description = [{
    Syntax:

    ```
    list-type ::= `list` `<` type `>`
    ```

    The value of `list` type represents a list number with a parameterized
    element type, which is composed of a real and imaginary value of that
    element type. The element must be a floating point or integer scalar type.

    Examples:

    ```mlir
    list<mutez>
    ```
  }];
  let parameters = (ins Michelson_Operation:$elementType);

  /*
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
  */

  let skipDefaultBuilders = 1;
  // let genVerifyDecl = 1;
    let assemblyFormat = [{
    `<` $elementType `>`
  }];

}

def Michelson_Pair: Michelson_Type<"Pair", "pair"> {
  let summary = "Michelson Pair with two parameterized element types";
  let description = [{
    Syntax:

    ```
    pair-type ::= `pair` `<` type `,` type `>`
    ```

    The value of `pair` type represents a pair of two parameterized
    element types. 

    Examples:

    ```mlir
    pair<mutez, mutez>
    ```
  }];

  // FIXME: polymorphic types
  let parameters = (ins Michelson_Mutez:$firstElementType, Michelson_Mutez:$secondElementType);

  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    `<` $firstElementType `,` $secondElementType `>`
  }];
}


def Michelson_Type : AnyTypeOf<[Michelson_PrimitiveType, Michelson_StructType]>;

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
