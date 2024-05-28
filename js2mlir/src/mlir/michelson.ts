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
// }}

// michelson ops {{
export class GetAmount extends Op {
  constructor(readonly ret: Value) {
    super();
  }

  toString() {
    return `${this.ret} = michelson.get_amount() : ${MutezType}`;
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
    return `${this.ret} = michelson.make_list() : ${ListType(this.type)}`;
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
    return `${this.ret} = michelson.make_pair(%${this.first}, %${this.second}) : ${PairType(this.firstType, this.secondType)}`;
  }
}
// }}
