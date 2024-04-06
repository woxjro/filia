// RUN: script-opt %s | script-opt | FileCheck %s

module {
    // CHECK-LABEL: func.func @smart_contract()
    func.func @smart_contract() {
        // CHECK: %{{.*}} = michelson.get_amount: !michelson.mutez
        %res = michelson.get_amount: !michelson.mutez
        return
    }
}
