#include <stdio.h>
#include <stdlib.h>

#include "fmi4c.h"
#include "fmi4c_lsdae.h"

#ifndef SIMPLE_DAE_FMU_PATH
#error "SIMPLE_DAE_FMU_PATH must be defined by the build system"
#endif

static void print_message(const char *msg) {
    printf("[fmi4c] %s\n", msg);
}

static const char *dep_kind_str(fmi3DependencyKind k)
{
    switch (k) {
        case fmi3Independent: return "independent";
        case fmi3Constant:    return "constant";
        case fmi3Fixed:       return "fixed";
        case fmi3Tunable:     return "tunable";
        case fmi3Discrete:    return "discrete";
        case fmi3Dependent:   return "dependent";
        default:              return "unknown";
    }
}

static void print_structure_entry(const char *label, fmiLsDaeModelStructureHandle *h)
{
    int n = fmiLsDae_getNumberOfDependencies(h);
    printf("  %s vr=%-12u  deps=%d", label, fmiLsDae_getValueReference(h), n);
    if (n > 0) {
        fmi3ValueReference deps[64];
        fmi3DependencyKind kinds[64];
        int take = (n < 64) ? n : 64;
        fmiLsDae_getDependencies(h, deps, take);
        printf(" [");
        for (int i = 0; i < take; ++i)
            printf("%s%u", i ? " " : "", deps[i]);
        printf("]");
        if (fmiLsDae_dependencyKindsDefined(h)) {
            fmiLsDae_getDependencyKinds(h, kinds, take);
            printf(" kinds=[");
            for (int i = 0; i < take; ++i)
                printf("%s%s", i ? " " : "", dep_kind_str(kinds[i]));
            printf("]");
        }
    }
    printf("\n");
}

int main(void)
{
    fmi4c_setMessageFunction(print_message);

    printf("Loading FMU: %s\n", SIMPLE_DAE_FMU_PATH);
    fmuHandle *fmu = fmi4c_loadFmu(SIMPLE_DAE_FMU_PATH, "SimpleDAE");
    if (!fmu) {
        fprintf(stderr, "ERROR: fmi4c_loadFmu failed\n");
        return EXIT_FAILURE;
    }

    printf("FMI version  : %d\n", (int)fmi4c_getFmiVersion(fmu));
    printf("Model name   : %s\n", fmi3_modelName(fmu));

    /* ---- fmi-ls-dae ---- */
    if (!fmiLsDae_isPresent(fmu)) {
        fprintf(stderr, "ERROR: fmi-ls-dae manifest not found in FMU\n");
        fmi4c_freeFmu(fmu);
        return EXIT_FAILURE;
    }

    printf("\n--- fmi-ls-dae manifest ---\n");
    printf("Name        : %s\n", fmiLsDae_getName(fmu));
    printf("Version     : %s\n", fmiLsDae_getVersion(fmu));
    printf("Description : %s\n", fmiLsDae_getDescription(fmu));

    printf("\nAlgebraicVariables (%d):\n", fmiLsDae_getNumberOfAlgebraicVariables(fmu));
    for (int i = 0; i < fmiLsDae_getNumberOfAlgebraicVariables(fmu); ++i) {
        fmiLsDaeAlgebraicVariableHandle *v = fmiLsDae_getAlgebraicVariableByIndex(fmu, i);
        printf("  [%d] vr=%u\n", i, fmiLsDae_getAlgebraicVariableValueReference(v));
    }

    printf("\nModelStructure - ContinuousStateDerivatives (%d):\n",
           fmiLsDae_getNumberOfContinuousStateDerivatives(fmu));
    for (int i = 0; i < fmiLsDae_getNumberOfContinuousStateDerivatives(fmu); ++i)
        print_structure_entry("CSD", fmiLsDae_getContinuousStateDerivativeByIndex(fmu, i));

    printf("\nModelStructure - Residuals (%d):\n",
           fmiLsDae_getNumberOfResiduals(fmu));
    for (int i = 0; i < fmiLsDae_getNumberOfResiduals(fmu); ++i)
        print_structure_entry("Res", fmiLsDae_getResidualByIndex(fmu, i));

    printf("\nModelStructure - Outputs (%d):\n",
           fmiLsDae_getNumberOfOutputs(fmu));
    for (int i = 0; i < fmiLsDae_getNumberOfOutputs(fmu); ++i)
        print_structure_entry("Out", fmiLsDae_getOutputByIndex(fmu, i));

    fmi4c_freeFmu(fmu);
    printf("\nOK\n");
    return EXIT_SUCCESS;
}
