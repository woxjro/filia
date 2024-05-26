import { Op, TypeAttr, Value, ppCommas } from '../mlir'
// import * as mlir from '../mlir'

// michelson {{
export const MutezType : TypeAttr = {
  toString() {
    return "!michelson.mutez"
  }
}

export const OperationType : TypeAttr = {
  toString() {
    return "!michelson.operation"
  }
}

export function ListType(type: TypeAttr) : TypeAttr {
  return {
    toString() {
      return `!michelson.list<${type}>`
    }
  }
}

export function PairType(first: TypeAttr, second: TypeAttr) : TypeAttr {
  return {
    toString() {
      return `!michelson.pair<${first},${second}>`
    }
  }
}
// }}

// michelson ops {{
export class GetAmount extends Op {
    constructor(readonly ret: Value) {
        super()
    }
    
    toString() {
        return `${this.ret} = michelson.get_amount() : ${MutezType}`
    }
}
// }}
