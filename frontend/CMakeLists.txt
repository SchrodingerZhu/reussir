function(add_frontend_target target_name)
    file(GLOB_RECURSE _hs_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hs"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cabal"
    )
    list(APPEND _hs_sources "${CMAKE_SOURCE_DIR}/cabal.project")

    set(_output "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target_name}${CMAKE_EXECUTABLE_SUFFIX}")

    add_custom_command(
        OUTPUT "${_output}"
        COMMAND ${CMAKE_COMMAND} -E env REUSSIR_PROJECT_ROOT=${CMAKE_SOURCE_DIR}
            cabal install -j ${target_name}
            --installdir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            --install-method=copy
            --overwrite-policy=always
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS ${_hs_sources} $<TARGET_FILE:MLIRReussirBridge>
        COMMENT "Building ${target_name} via cabal"
        USES_TERMINAL
    )

    add_custom_target(${target_name} DEPENDS "${_output}")
endfunction()

add_frontend_target(reussir-elab)
add_frontend_target(reussir-compiler)
add_frontend_target(reussir-repl)
add_frontend_target(reussir-parser)
add_frontend_target(reussir-surface-verify)
add_frontend_target(reussir-lsp)
