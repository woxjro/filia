// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Michelson/MichelsonDialect.h"
#include "Michelson/MichelsonOps.h"
#include "Michelson/MichelsonTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::michelson;

#include "Michelson/MichelsonOpsDialect.cpp.inc"

namespace mlir {
namespace michelson {
namespace detail {
    struct StructTypeStorage : public mlir::TypeStorage {
      /// The `KeyTy` is a required type that provides an interface for the storage
      /// instance. This type will be used when uniquing an instance of the type
      /// storage. For our struct type, we will unique each instance structurally on
      /// the elements that it contains.
      using KeyTy = llvm::ArrayRef<mlir::Type>;

      /// A constructor for the type storage instance.
      StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
          : elementTypes(elementTypes) {}

      /// Define the comparison function for the key type with the current storage
      /// instance. This is used when constructing a new instance to ensure that we
      /// haven't already uniqued an instance of the given key.
      bool operator==(const KeyTy &key) const { return key == elementTypes; }

      /// Define a hash function for the key type. This is used when uniquing
      /// instances of the storage.
      /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
      /// have hash functions available, so we could just omit this entirely.
      static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key);
      }

      /// Define a construction function for the key type from a set of parameters.
      /// These parameters will be provided when constructing the storage instance
      /// itself, see the `StructType::get` method further below.
      /// Note: This method isn't necessary because KeyTy can be directly
      /// constructed with the given parameters.
      static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
        return KeyTy(elementTypes);
      }

      /// Define a construction method for creating a new instance of this storage.
      /// This method takes an instance of a storage allocator, and an instance of a
      /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
      /// allocations used to create the type storage and its internal.
      static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
        // Copy the elements from the provided `KeyTy` into the allocator.
        llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

        // Allocate the storage instance and construct it.
        return new (allocator.allocate<StructTypeStorage>())
            StructTypeStorage(elementTypes);
      }

      /// The following field contains the element types of the struct.
      llvm::ArrayRef<mlir::Type> elementTypes;
    };

    /// This class defines the Toy struct type. It represents a collection of
    /// element types. All derived types in MLIR must inherit from the CRTP class
    /// 'Type::TypeBase'. It takes as template parameters the concrete type
    /// (StructType), the base class to use (Type), and the storage class
    /// (StructTypeStorage).
    class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                                   StructTypeStorage> {
    public:
      /// Inherit some necessary constructors from 'TypeBase'.
      using Base::Base;

      /// Create an instance of a `StructType` with the given element types. There
      /// *must* be at least one element type.
      static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
        assert(!elementTypes.empty() && "expected at least 1 element type");

        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. The first parameter is the context to unique in. The
        // parameters after are forwarded to the storage instance.
        mlir::MLIRContext *ctx = elementTypes.front().getContext();
        return Base::get(ctx, elementTypes);
      }

      /// Returns the element types of this struct type.
      llvm::ArrayRef<mlir::Type> getElementTypes() {
        // 'getImpl' returns a pointer to the internal storage instance.
        return getImpl()->elementTypes;
      }

      /// Returns the number of element type held by this struct.
      size_t getNumElementTypes() { return getElementTypes().size(); }
    };

} // namespace detail
} // namespace michelson
} // namespace mlir


void MichelsonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Michelson/MichelsonOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Michelson/MichelsonOpsTypes.cpp.inc"
      >();
  addTypes<mlir::michelson::detail::StructType>();

}
