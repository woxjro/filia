set(LLVM_TARGET_DEFINITIONS MichelsonOps.td)
mlir_tablegen(MichelsonOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(MichelsonOpsEnums.cpp.inc -gen-enum-defs)

add_mlir_dialect(MichelsonOps michelson)
add_mlir_doc(MichelsonDialect MichelsonDialect Michelson/ -gen-dialect-doc)
add_mlir_doc(MichelsonOps MichelsonOps Michelson/ -gen-op-doc)
