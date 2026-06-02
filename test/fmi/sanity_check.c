#include <stdio.h>
#include <stdlib.h>

#include "fmi4c.h"

static void print_message(const char *msg) {
    printf("[fmi4c] %s\n", msg);
}

int main(void) {
    fmi4c_setMessageFunction(print_message);

    printf("Loading FMU: %s\n", FMU_PATH);

    fmuHandle *fmu = fmi4c_loadFmu(FMU_PATH, "BouncingBall");
    if (!fmu) {
        fprintf(stderr, "ERROR: fmi4c_loadFmu failed\n");
        return EXIT_FAILURE;
    }

    fmiVersion_t version = fmi4c_getFmiVersion(fmu);
    if (version != fmiVersion3) {
        fprintf(stderr, "ERROR: expected FMI 3.0, got version enum %d\n", (int)version);
        fmi4c_freeFmu(fmu);
        return EXIT_FAILURE;
    }
    printf("FMI version : 3.0\n");

    printf("Model name  : %s\n", fmi3_modelName(fmu));
    printf("Description : %s\n", fmi3_description(fmu));
    printf("Gen. tool   : %s\n", fmi3_generationTool(fmu));

    printf("Supports ME : %s\n", fmi3_supportsModelExchange(fmu)      ? "yes" : "no");
    printf("Supports CS : %s\n", fmi3_supportsCoSimulation(fmu)       ? "yes" : "no");
    printf("Supports SE : %s\n", fmi3_supportsScheduledExecution(fmu) ? "yes" : "no");

    int n_vars = fmi3_getNumberOfVariables(fmu);
    printf("Variables   : %d\n", n_vars);

    for (int i = 1; i <= n_vars; i++) {
        fmi3VariableHandle *var = fmi3_getVariableByIndex(fmu, i);
        printf("  [%d] %s\n", i, fmi3_getVariableName(var));
    }

    fmi4c_freeFmu(fmu);
    printf("OK\n");
    return EXIT_SUCCESS;
}
