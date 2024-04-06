// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef MICHELSON_DIALECT
#define MICHELSON_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Michelson dialect definition.
//===----------------------------------------------------------------------===//

def Michelson_Dialect : Dialect {
    let name ="michelson";
    let summary = "A MLIR dialect for Michelson.";
    let description = [{
        This dialect captures operations needed to encode Michelson programs
        in MLIR.
    }];
    let cppNamespace = "::mlir::michelson";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base michelson operation definition.
//===----------------------------------------------------------------------===//

class Michelson_Op<string mnemonic, list<Trait> traits = []> :
        Op<Michelson_Dialect, mnemonic, traits>;

#endif // MICHELSON_DIALECT
