function MichelsonGetAmount() {
  return 100;
}

function MichelsonMakePair(a, b) {
  return [a, b];
}

function MichelsonMakeOperationList() {
  return [];
}

function MichelsonMakeResultPair(a, b) {
    return [a, b];
}


/* 
 * @storage {mutez} storage
 * @param {mutez} param
 */
function smartContract(storage, param) {
    const amount = MichelsonGetAmount();
    const operations = MichelsonMakeOperationList();
    const pair = MichelsonMakeResultPair(operations, amount);
    return pair;
}
