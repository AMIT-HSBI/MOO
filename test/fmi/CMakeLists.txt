# FMI sanity check
add_executable(fmi_sanity_check sanity_check.c)

target_link_libraries(fmi_sanity_check PRIVATE fmi4c)

target_compile_definitions(fmi_sanity_check PRIVATE
    FMU_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/BouncingBall.fmu"
)

add_test(NAME fmi_sanity_check COMMAND fmi_sanity_check)

# fmi-ls-dae sanity check - references an FMU that is not redistributable,
# so the path is supplied via the SIMPLE_DAE_FMU environment variable.
if(DEFINED ENV{SIMPLE_DAE_FMU})
    add_executable(lsdae_sanity_check lsdae_sanity_check.c)
    target_link_libraries(lsdae_sanity_check PRIVATE fmi4c)
    target_compile_definitions(lsdae_sanity_check PRIVATE
        SIMPLE_DAE_FMU_PATH="$ENV{SIMPLE_DAE_FMU}"
    )
    add_test(NAME lsdae_sanity_check COMMAND lsdae_sanity_check)
else()
    message(WARNING "SIMPLE_DAE_FMU not set - skipping lsdae_sanity_check")
endif()

# FMI ls-dae test 1
add_executable(ls_dae_fourbar ls_dae_fourbar.cpp)

target_link_libraries(ls_dae_fourbar PRIVATE fmi4c moo)

target_compile_definitions(ls_dae_fourbar PRIVATE
    FOURBAR_FMU_PATH="$ENV{FOURBAR_FMU}"
)

add_test(NAME ls_dae_fourbar COMMAND ls_dae_fourbar)

# FMI ls-dae test 2
add_executable(ls_dae_simple_dae ls_dae_simple_dae.cpp)

target_link_libraries(ls_dae_simple_dae PRIVATE fmi4c moo)

target_compile_definitions(ls_dae_simple_dae PRIVATE
    SIMPLE_DAE_FMU_PATH="$ENV{SIMPLE_DAE_FMU}"
)

add_test(NAME ls_dae_simple_dae COMMAND ls_dae_simple_dae)

# FMI ls-dae test 3
add_executable(ls_dae_crane ls_dae_crane.cpp)

target_link_libraries(ls_dae_crane PRIVATE fmi4c moo)

target_compile_definitions(ls_dae_crane PRIVATE
    CRANE_FMU_PATH="$ENV{CRANE_FMU}")

add_test(NAME ls_dae_crane COMMAND ls_dae_crane)

# FMI ls-dae test 4 - DAE Multilink
add_executable(ls_dae_multilink ls_dae_multilink.cpp)

target_link_libraries(ls_dae_multilink PRIVATE fmi4c moo)

target_compile_definitions(ls_dae_multilink PRIVATE
    MULTILINK_FMU_PATH="$ENV{MULTILINK_FMU}"
)

add_test(NAME ls_dae_multilink COMMAND ls_dae_multilink)

# FMI ls-dae test 5 - ODE Multilink
add_executable(ls_dae_multilink_ode ls_dae_multilink_ode.cpp)

target_link_libraries(ls_dae_multilink_ode PRIVATE fmi4c moo)

target_compile_definitions(ls_dae_multilink_ode PRIVATE
    MULTILINK_ODE_FMU_PATH="$ENV{MULTILINK_ODE_FMU}"
)

add_test(NAME ls_dae_multilink_ode COMMAND ls_dae_multilink_ode)
