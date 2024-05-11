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

namespace detail {
    struct StructTypeStorage;

    /// This class defines the Toy struct type. It represents a collection of
    /// element types. All derived types in MLIR must inherit from the CRTP class
    /// 'Type::TypeBase'. It takes as template parameters the concrete type
    /// (StructType), the base class to use (Type), and the storage class
    /// (StructTypeStorage).
    class StructType;
}
} // namespace michelson
} // namespace mlir


#define GET_TYPEDEF_CLASSES
#include "Michelson/MichelsonOpsTypes.h.inc"

#endif // MLIR_DIALECT_MICHELSON_IR_MICHELSONTYPES_H_
