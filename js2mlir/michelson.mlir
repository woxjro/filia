func.func @smart_contract(param: !michelson.mutez, storage: !michelson.mutez) -> (!michelson.pair<!michelson.list<!michelson.operation>,!michelson.mutez>) {
  %locals = js.empty_locals : !js.locals
  %0 = michelson.get_amount() : !michelson.mutez
  %locals.0 = js.local_decl %locals, const, amount, %0 : !js.locals
  %1 = michelson.make_list() : !michelson.list<!michelson.operation>
  %locals.1 = js.local_decl %locals.0, const, operations, %1 : !js.locals
  %2 = michelson.make_pair(%operations, %amount) : !michelson.pair<!michelson.list<!michelson.operation>,!michelson.mutez>
  %locals.2 = js.local_decl %locals.1, const, pair, %2 : !js.locals
  %pair = js.local_get %locals.2, "pair" : !js.locals
  return %pair : !js.value
}
