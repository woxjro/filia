#set(CMAKE_CXX_STANDARD 17)

add_mlir_dialect_library(MLIRMichelson
        MichelsonDialect.cpp
        MichelsonOps.cpp
        MichelsonTypes.cpp
        FunctionValueTransitionMap.cpp
        MichelsonDomain.cpp
        MichelsonLoadStorePass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Michelson

        DEPENDS
        MLIRMichelsonOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
target_compile_features(obj.MLIRMichelson PRIVATE cxx_std_17)
