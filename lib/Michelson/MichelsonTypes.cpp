// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Michelson/MichelsonDialect.h"
#include "Michelson/MichelsonTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

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


namespace mlir {
namespace michelson {
namespace detail {
    /*
    struct ListTypeStorage : public mlir::TypeStorage {
     
      /// The `KeyTy` is a required type that provides an interface for the storage
      /// instance. This type will be used when uniquing an instance of the type
      /// storage. For our struct type, we will unique each instance structurally on
      /// the elements that it contains.
      using KeyTy = mlir::Type;

      /// A constructor for the type storage instance.
      ListTypeStorage(mlir::Type elementType)
          : elementType(elementType) {}

      /// Define the comparison function for the key type with the current storage
      /// instance. This is used when constructing a new instance to ensure that we
      /// haven't already uniqued an instance of the given key.
      bool operator==(const KeyTy &key) const { return key == elementType; }

      /// Define a hash function for the key type. This is used when uniquing
      /// instances of the storage, see the `ListType::get` method.
      /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
      /// have hash functions available, so we could just omit this entirely.
      static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key);
      }

      /// Define a construction function for the key type from a set of parameters.
      /// These parameters will be provided when constructing the storage instance
      /// itself.
      /// Note: This method isn't necessary because KeyTy can be directly
      /// constructed with the given parameters.
      static KeyTy getKey(mlir::Type elementType) {
        return KeyTy(elementType);
      }

      /// Define a construction method for creating a new instance of this storage.
      /// This method takes an instance of a storage allocator, and an instance of a
      /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
      /// allocations used to create the type storage and its internal.
      static ListTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
        // Copy the elements from the provided `KeyTy` into the allocator.
        mlir::Type elementType = allocator.copyInto(key);

        // Allocate the storage instance and construct it.
        return new (allocator.allocate<ListTypeStorage>())
            ListTypeStorage(elementType);
      }

      /// The following field contains the element types of the struct.
      mlir::Type elementType;
    };
    */
} // namespace detail
} // namespace michelson
} // namespace mlir
