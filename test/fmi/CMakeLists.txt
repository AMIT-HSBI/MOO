# FMI sanity check
add_executable(fmi_sanity_check sanity_check.c)

target_link_libraries(fmi_sanity_check PRIVATE fmi4c)

target_compile_definitions(fmi_sanity_check PRIVATE
    FMU_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/BouncingBall.fmu"
)

add_test(NAME fmi_sanity_check COMMAND fmi_sanity_check)

function(add_fmu_test target source env_var macro_name)
    if(DEFINED ENV{${env_var}})
        add_executable(${target} ${source})

        target_link_libraries(${target} PRIVATE fmi4c moo)

        target_compile_definitions(${target} PRIVATE
            ${macro_name}="$ENV{${env_var}}"
        )

        add_test(NAME ${target} COMMAND ${target})
    else()
        message(STATUS "${env_var} not set - skipping ${target}")
    endif()
endfunction()

# FMI-LS-DAE sanity check
add_fmu_test(
    lsdae_sanity_check
    lsdae_sanity_check.c
    SIMPLE_DAE_FMU
    SIMPLE_DAE_FMU_PATH
)

# FMI-LS-DAE examples
add_fmu_test(
    ls_dae_fourbar
    ls_dae_fourbar.cpp
    FOURBAR_FMU
    FOURBAR_FMU_PATH
)

add_fmu_test(
    ls_dae_simple_dae
    ls_dae_simple_dae.cpp
    SIMPLE_DAE_FMU
    SIMPLE_DAE_FMU_PATH
)

add_fmu_test(
    ls_dae_crane
    ls_dae_crane.cpp
    CRANE_FMU
    CRANE_FMU_PATH
)

add_fmu_test(
    ls_dae_multilink
    ls_dae_multilink.cpp
    MULTILINK_FMU
    MULTILINK_FMU_PATH
)

add_fmu_test(
    ls_dae_multilink_ode
    ls_dae_multilink_ode.cpp
    MULTILINK_ODE_FMU
    MULTILINK_ODE_FMU_PATH
)
