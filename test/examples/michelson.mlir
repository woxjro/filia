// RUN: script-opt %s | script-opt | FileCheck %s

module {
    // CHECK-LABEL: func.func @smart_contract()
    func.func @smart_contract() {
        // CHECK: %{{.*}} = michelson.get_amount: !michelson.mutez
        %amount1 = michelson.get_amount(): !michelson.mutez
        %amount2 = michelson.get_amount(): !michelson.mutez
        %list = michelson.make_list(): !michelson.list<!michelson.operation>
        %pair = michelson.make_pair(%amount1, %amount2): !michelson.pair<!michelson.mutez, !michelson.mutez>
        return
    }
}
