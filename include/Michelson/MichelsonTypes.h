#ifndef MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES_H_
#define MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES_H_

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Michelson Dialect Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace michelson {
/// This class represents the base class of all Michelson types.
class MichelsonType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);
};
} // namespace michelson
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "Michelson/MichelsonOpsTypes.h.inc"

#endif // MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES_H_