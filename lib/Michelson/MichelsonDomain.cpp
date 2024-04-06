#include "MichelsonDomain.h"
#include "ValueTranslator.h"

using namespace llvm;

CellDomain CellDomain::initializeFromPrev(
    ValueTranslator& translator,
    const CellDomain& srcDomain,
    const mlir::Value& tgtCell) {
  switch (srcDomain.status) {
  case CellDomain::EMPTY:
    return CellDomain::empty();
  case CellDomain::CELL_UNKNOWN:
    return CellDomain::unknown();
  case CellDomain::VALUE:
    switch (srcDomain._value.type) {
    case ValueDomain::VALUE:
      if (auto newV = translator.valueToTarget(srcDomain._value.value)) {
        return CellDomain::value(ValueDomain::make_value(newV));
      } else {
        unsigned argIndex = translator.getCellValueArg(tgtCell);
        return CellDomain::value(ValueDomain::make_argument(argIndex));
      }
      break;
    case ValueDomain::BUILTIN:
    case ValueDomain::MODULE:
      return CellDomain::value(ValueDomain(srcDomain._value));
      break;
    case ValueDomain::ARGUMENT:
      {
        unsigned argIndex = translator.getCellValueArg(tgtCell);
        return CellDomain::value(ValueDomain::make_argument(argIndex));
      }
      break;
    //default:
    //  report_fatal_error("Invalid value domain");
    }
  //default:
  //  report_fatal_error("Invalid cell domain");
  }
}

bool CellDomain::mergeFromPrev(ValueTranslator& translator, const CellDomain& srcDomain, const mlir::Value& tgtCell) {
  switch (status) {
  case CellDomain::CELL_UNKNOWN:
    return false;
  case CellDomain::EMPTY:
    if (srcDomain.status != CellDomain::EMPTY) {
      status = CELL_UNKNOWN;
      return true;
    }
    return false;
  case CellDomain::VALUE:
    if (srcDomain.status != CellDomain::VALUE) {
      status = CELL_UNKNOWN;
      return true;
    } else {
      const auto& newDomain = srcDomain._value;
      if (_value.type == ValueDomain::ARGUMENT) {
        // Do nothing
        return false;
      } else if (_value == newDomain) {
        // Do nothing
        return false;
      } else {
        unsigned argIndex = translator.getCellValueArg(tgtCell);
        _value = ValueDomain::make_argument(argIndex);
        return true;
      }
    }
  }
}


/*
void ScopeDomain::addBuiltins() {
  addBuiltin(::mlir::michelson::BuiltinAttr::abs);
  addBuiltin(::mlir::michelson::BuiltinAttr::aiter);
  addBuiltin(::mlir::michelson::BuiltinAttr::all);
  addBuiltin(::mlir::michelson::BuiltinAttr::any);
  addBuiltin(::mlir::michelson::BuiltinAttr::anext);
  addBuiltin(::mlir::michelson::BuiltinAttr::ascii);
  addBuiltin(::mlir::michelson::BuiltinAttr::bin);
  addBuiltin(::mlir::michelson::BuiltinAttr::bool_builtin, "bool");
  addBuiltin(::mlir::michelson::BuiltinAttr::breakpoint);
  addBuiltin(::mlir::michelson::BuiltinAttr::bytearray);
  addBuiltin(::mlir::michelson::BuiltinAttr::bytes);
  addBuiltin(::mlir::michelson::BuiltinAttr::callable);
  addBuiltin(::mlir::michelson::BuiltinAttr::chr);
  addBuiltin(::mlir::michelson::BuiltinAttr::classmethod);
  addBuiltin(::mlir::michelson::BuiltinAttr::compile);
  addBuiltin(::mlir::michelson::BuiltinAttr::complex);
  addBuiltin(::mlir::michelson::BuiltinAttr::delattr);
  addBuiltin(::mlir::michelson::BuiltinAttr::dict);
  addBuiltin(::mlir::michelson::BuiltinAttr::dir);
  addBuiltin(::mlir::michelson::BuiltinAttr::divmod);
  addBuiltin(::mlir::michelson::BuiltinAttr::enumerate);
  addBuiltin(::mlir::michelson::BuiltinAttr::eval);
  addBuiltin(::mlir::michelson::BuiltinAttr::exec);
  addBuiltin(::mlir::michelson::BuiltinAttr::filter);
  addBuiltin(::mlir::michelson::BuiltinAttr::float_builtin, "float");
  addBuiltin(::mlir::michelson::BuiltinAttr::format);
  addBuiltin(::mlir::michelson::BuiltinAttr::frozenset);
  addBuiltin(::mlir::michelson::BuiltinAttr::getattr);
  addBuiltin(::mlir::michelson::BuiltinAttr::globals);
  addBuiltin(::mlir::michelson::BuiltinAttr::hasattr);
  addBuiltin(::mlir::michelson::BuiltinAttr::hash);
  addBuiltin(::mlir::michelson::BuiltinAttr::help);
  addBuiltin(::mlir::michelson::BuiltinAttr::hex);
  addBuiltin(::mlir::michelson::BuiltinAttr::id);
  addBuiltin(::mlir::michelson::BuiltinAttr::input);
  addBuiltin(::mlir::michelson::BuiltinAttr::int_builtin, "int");
  addBuiltin(::mlir::michelson::BuiltinAttr::isinstance);
  addBuiltin(::mlir::michelson::BuiltinAttr::issubclass);
  addBuiltin(::mlir::michelson::BuiltinAttr::iter);
  addBuiltin(::mlir::michelson::BuiltinAttr::len);
  addBuiltin(::mlir::michelson::BuiltinAttr::list);
  addBuiltin(::mlir::michelson::BuiltinAttr::locals);
  addBuiltin(::mlir::michelson::BuiltinAttr::map);
  addBuiltin(::mlir::michelson::BuiltinAttr::max);
  addBuiltin(::mlir::michelson::BuiltinAttr::memoryview);
  addBuiltin(::mlir::michelson::BuiltinAttr::min);
  addBuiltin(::mlir::michelson::BuiltinAttr::next);
  addBuiltin(::mlir::michelson::BuiltinAttr::object);
  addBuiltin(::mlir::michelson::BuiltinAttr::oct);
  addBuiltin(::mlir::michelson::BuiltinAttr::open);
  addBuiltin(::mlir::michelson::BuiltinAttr::ord);
  addBuiltin(::mlir::michelson::BuiltinAttr::pow);
  addBuiltin(::mlir::michelson::BuiltinAttr::print);
  addBuiltin(::mlir::michelson::BuiltinAttr::property);
  addBuiltin(::mlir::michelson::BuiltinAttr::range);
  addBuiltin(::mlir::michelson::BuiltinAttr::repr);
  addBuiltin(::mlir::michelson::BuiltinAttr::reversed);
  addBuiltin(::mlir::michelson::BuiltinAttr::round);
  addBuiltin(::mlir::michelson::BuiltinAttr::set);
  addBuiltin(::mlir::michelson::BuiltinAttr::setattr);
  addBuiltin(::mlir::michelson::BuiltinAttr::slice);
  addBuiltin(::mlir::michelson::BuiltinAttr::sorted);
  addBuiltin(::mlir::michelson::BuiltinAttr::staticmethod);
  addBuiltin(::mlir::michelson::BuiltinAttr::str);
  addBuiltin(::mlir::michelson::BuiltinAttr::sum);
  addBuiltin(::mlir::michelson::BuiltinAttr::super);
  addBuiltin(::mlir::michelson::BuiltinAttr::tuple);
  addBuiltin(::mlir::michelson::BuiltinAttr::type);
  addBuiltin(::mlir::michelson::BuiltinAttr::vars);
  addBuiltin(::mlir::michelson::BuiltinAttr::zip);
  addBuiltin(::mlir::michelson::BuiltinAttr::import, "__import__");
}
*/

// Create a locals domain from a previous block
void LocalsDomain::populateFromPrev(ValueTranslator& translator, const LocalsDomain& prev) {
  // Iterate through all values in previous block.
  for (auto i = prev.cellValues.begin(); i != prev.cellValues.end(); ++i) {
    auto cell = i->first;
    auto& srcDomain = i->second;
    auto tgtCell = translator.valueToTarget(cell);
    if (!tgtCell) {
      std::string str;
      mlir::AsmState state(block->getParentOp());
      llvm::raw_string_ostream o(str);
      o << "Error in block ";
      block->printAsOperand(o);
      o << ": Dropping cell ";
      cell.printAsOperand(o, state);
      o << ".";
      report_fatal_error(str.c_str());

//      continue;

    }
    auto p = cellValues.try_emplace(tgtCell,
      CellDomain::initializeFromPrev(translator, srcDomain, tgtCell));
    if (!p.second) {
      report_fatal_error("Translator maps two values to single value.");
    }
  }
}

/**
 * `x.mergeFromPrev(trans, prev) applies the translator and takes the
 * domain containing facts true in both x and trans(prev).
 */
bool LocalsDomain::mergeFromPrev(ValueTranslator& translator, const LocalsDomain& prev) {
  // Create set containing all the cells in this block that have associated domains.
  llvm::DenseSet<mlir::Value> seen;

  // Propagate each constraint in previous block to this  block.
  bool changed = false;
  for (auto i = prev.cellValues.begin(); i != prev.cellValues.end(); ++i) {
    auto prevCell = i->first;
    auto& srcDomain = i->second;
    auto tgtCell = translator.valueToTarget(prevCell);
    if (!tgtCell)
      continue;
    // Lookup domain for target value in this set.
    auto prevTgtIter = this->cellValues.find(tgtCell);
    if (prevTgtIter == this->cellValues.end())
      continue;
    if (!seen.insert(tgtCell).second) {
      report_fatal_error("Duplicate values in scope.");
    }
    if (prevTgtIter->second.mergeFromPrev(translator, srcDomain, tgtCell))
      changed = true;
  }


  // Remove all cells that were unconstrained in prev (and thus still in unseen)
  for (auto i = cellValues.begin(); i != cellValues.end(); ++i) {
    if (!i->second.is_unknown() && !seen.contains(i->first)) {
      i->second = CellDomain::unknown();
      changed = true;
    }
  }

  return changed;
}
