#pragma once
#include <mlir/Parser.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <llvm/ADT/ImmutableMap.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>

#include "Python/PythonDialect.h"
#include "Python/PythonOps.h"

inline
void fatal_error(const char* message) {
  fprintf(stderr, "%s\n", message);
  exit(-1);
}

class ValueTranslator;

/**
 *
 * Identifies information about a variable name stored in a scope.
 * Information is relative to a specific execution point in a block.
 *
 * There are currently three "types" of
 * * VALUE.  Indicates the name is associated with a specific value in the
 *   block.
 * * BUILTIN.  Indicates the name is associated with a builtin function.
 * * MODULE.  Indicates the name is associated with a specific module.
 * * ARGUMENT.  Indicates the name could be associated with an argument that
 *   could be added to the block.
 */
class ValueDomain {
public:
  enum ValueDomainType { VALUE, BUILTIN, MODULE, ARGUMENT };

  ValueDomainType type;
  // Value this is associated with.
  // (defined when type == VALUE)
  mlir::Value value;
  // Name of builtin or module
  // (defined when `type == BUILTIN || type == MODULE`).
  llvm::StringRef name;
  // Index of argument
  unsigned argument;

public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(type);
    switch (type) {
      case VALUE:
        ID.AddPointer(value.getImpl());
        break;
      case BUILTIN:
        ID.AddString(name);
        break;
      case MODULE:
        ID.AddString(name);
        break;
      case ARGUMENT:
        ID.AddInteger(argument);
        break;
    }
  }

  static
  ValueDomain make_value(const mlir::Value& value) {
    return { .type = VALUE, .value = value };
  }

  static
  ValueDomain make_builtin(llvm::StringRef name) {
    return { .type = BUILTIN, .name = name };
  }

  static
  ValueDomain make_module(llvm::StringRef name) {
    return { .type = MODULE, .name = name };
  }

  static
  ValueDomain make_argument(unsigned arg) {
    return { .type = ARGUMENT, .argument = arg };
  }

  mlir::Value getValue(mlir::OpBuilder& builder,
                       const mlir::Location& location,
                       const std::vector<mlir::Value>& argValues) const {
    switch (type) {
    case VALUE:
      return value;
    case BUILTIN:
      {
        auto b = builder.create<mlir::python::Builtin>(location, name);
        return b.result();
      }
    case MODULE:
      {
        auto b = builder.create<mlir::python::Module>(location, name);
        return b.result();
      }
    case ARGUMENT:
      return argValues[argument];
    }
    return mlir::Value();
  }
};

static inline
bool operator==(const ValueDomain& x, const ValueDomain& y) {
  if (x.type != y.type)
    return false;
  switch (x.type) {
  case ValueDomain::VALUE:
    return x.value == y.value;
  case ValueDomain::BUILTIN:
    return x.name == y.name;
  case ValueDomain::MODULE:
    return x.name == y.name;
  case ValueDomain::ARGUMENT:
    return x.argument == y.argument;
  }
}

/**
 * This is the abstract domain that we associate with a scope.
 */
class ScopeDomain {
private:

  void addBinding(llvm::StringRef name, const ValueDomain& v) {
    map.insert(std::make_pair(name, v));
  }

  void addBuiltin(llvm::StringRef name) {
    addBinding(name, ValueDomain::make_builtin(name));
  }

  llvm::DenseMap<llvm::StringRef, ValueDomain> map;
public:

  explicit ScopeDomain() {
  }

  /**
   * initializeFromPrev initializes a scope domain for a target block using information from the scope
   * domain in a previous block.  This may need to requst the translator generate block
   * arguments to pass values from the source to the target.
   *
   * @param translator Provides functionality for mapping values into target block
   * @param srcDomain Source domain to pull constraints from.
   * @param tgtValue Value denoting scope in target block.
   */
  void initializeFromPrev(ValueTranslator& translator, const ScopeDomain& srcDomain, const mlir::Value& tgtValue);

  /**
   * mergeFromPrev updates scope constraints to reflect only constraints in both domains.
   *
   * @param translator Provides functionality for mapping values into target block
   * @param srcDomain Source domain to pull constraints from.
   * @param tgtValue Value denoting scope in target block.
   */
  bool mergeFromPrev(ValueTranslator& translator, const ScopeDomain& srcDomain, const mlir::Value& tgtValue);

  /**
   * Add mappings from builtin names to the corresponding builtin function.
   */
  void addBuiltins() {
    addBuiltin("eval");
    addBuiltin("getattr");
    addBuiltin("isinstance");
    addBuiltin("open");
    addBuiltin("print");
    addBuiltin("slice");
  }

  void import(const llvm::StringRef& module, const llvm::StringRef& asName) {
    addBinding(asName, ValueDomain::make_module(module));
  }

  void setValue(const llvm::StringRef& name, mlir::Value value) {
    addBinding(name, ValueDomain::make_value(value));
  }

  ValueDomain* getValue(const llvm::StringRef& name) {
    auto i = map.find(name);
    return (i != map.end()) ? &i->second : 0;
  }

  const ValueDomain* getValue(const llvm::StringRef& name) const {
    auto i = map.find(name);
    return (i != map.end()) ? &i->second : 0;
  }
};

/**
 * This maps values in the block for scopes to the abstract domain associated with them.
 */
class LocalsDomain {
  // Map values for scope variables to the associated domain.
  llvm::DenseMap<mlir::Value, ScopeDomain> scopeDomains;

public:

  LocalsDomain() {

  }

  // Create a locals domain from a previous block
  void populateFromPrev(ValueTranslator& translator, const LocalsDomain& prev);

  /**
   * `x.mergeFromPrev(trans, prev) applies the translator and takes the
   * domain containing facts true in both x and trans(prev).
   */
  bool mergeFromPrev(ValueTranslator& translator, const LocalsDomain& prev);

  ScopeDomain& scope_domain(const mlir::Value& v) {
    auto i = scopeDomains.find(v);
    if (i != scopeDomains.end())
      return i->second;

    ScopeDomain d;
    auto r = scopeDomains.insert(std::make_pair(v, d));
    return r.first->second;
  }

  void scope_init(mlir::python::ScopeInit op) {
    ScopeDomain d;
    d.addBuiltins();
    scopeDomains.insert(std::make_pair(op.getResult(), d));
  }

  void scope_extend(mlir::python::ScopeExtend op) {
    ScopeDomain d;
    scopeDomains.insert(std::make_pair(op.getResult(), d));
  }

  void scope_import(mlir::python::ScopeImport op) {
    auto& m = scope_domain(op.scope());
    m.import(op.module(), op.asName());
  }

  ValueDomain* scope_get(mlir::python::ScopeGet op) {
    auto m = scope_domain(op.scope());
    return m.getValue(op.name());
  }

  void scope_set(mlir::python::ScopeSet op) {
    scope_domain(op.scope()).setValue(op.name(), op.value());
  }

  mlir::Value getScopeValue(mlir::Value scope,
                            llvm::StringRef name,
                            mlir::OpBuilder& builder,
                            const mlir::Location& location,
                            const std::vector<mlir::Value> &argValues) const {
    auto i = scopeDomains.find(scope);
    if (i == scopeDomains.end())
      fatal_error("Unknown scope in getScopeValue");
    auto v = i->second.getValue(name);
    if (!v) return mlir::Value();
    return v->getValue(builder, location, argValues);
  }
};