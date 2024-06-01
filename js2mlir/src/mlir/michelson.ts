import { Op, TypeAttr, Value, ppCommas } from '../mlir';
// import * as mlir from '../mlir'

// michelson {{
export const MutezType: TypeAttr = {
  toString() {
    return '!michelson.mutez';
  },
};

export const OperationType: TypeAttr = {
  toString() {
    return '!michelson.operation';
  },
};

export function ListType(type: TypeAttr): TypeAttr {
  return {
    toString() {
      return `!michelson.list<${type}>`;
    },
  };
}

export function PairType(first: TypeAttr, second: TypeAttr): TypeAttr {
  return {
    toString() {
      return `!michelson.pair<${first},${second}>`;
    },
  };
}

export function getTypeFromString(type: string): TypeAttr {
  switch (type) {
    case 'mutez':
      return MutezType;
    default:
      throw new Error(`Unsupported type: ${type}`);
  }
}

// }}

// michelson ops {{
export class GetAmount extends Op {
  constructor(readonly ret: Value) {
    super();
  }

  toString() {
    return `%${this.ret} = "michelson.get_amount"() : ${MutezType}`;
  }
}

export class MakeList extends Op {
  constructor(
    readonly ret: Value,
    readonly type: TypeAttr,
  ) {
    super();
  }

  toString() {
    return `%${this.ret} = "michelson.make_list"() : ${ListType(this.type)}`;
  }
}

export class MakePair extends Op {
  constructor(
    readonly ret: Value,
    readonly first: Value,
    readonly second: Value,
    readonly firstType: TypeAttr,
    readonly secondType: TypeAttr,
  ) {
    super();
  }

  toString() {
    return `%${this.ret} = "michelson.make_pair"(%${this.first}, %${this.second}) : (${this.firstType}, ${this.secondType}) -> ${PairType(this.firstType, this.secondType)}`;
  }
}
/*
func.func @smart_contract(%parameter: !michelson.unit, %storage: !michelson.unit)
    -> !michelson.pair<!michelson.list<!michelson.operation>, !michelson.unit> {

    %unit = "michelson.get_unit"() : () -> !michelson.unit
    %operations = "michelson.make_list"() : () -> !michelson.list<!michelson.operation>

    %res = "michelson.make_pair"(%operations, %unit) :
      (!michelson.list<!michelson.operation> , !michelson.unit)
        -> !michelson.pair<!michelson.list<!michelson.operation>, !michelson.unit>

    return %res : !michelson.pair<!michelson.list<!michelson.operation>, !michelson.unit>
  }
*/
// }}
