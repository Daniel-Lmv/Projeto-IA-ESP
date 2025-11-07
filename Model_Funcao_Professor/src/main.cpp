#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model.h"
#include "NNModel.h"

extern "C" void app_main(void){
    alignas(16) constexpr int arenaSize = 20*1024;
    NNModel *Model_Func = new NNModel(arenaSize, model);
    float input[5];
    float output;

    input[0] = 938.0f;
    input[1] = 125.0f;
    input[2] = 196.0f;
    input[3] = 55.0f;
    input[4] = 48.0f;

    while(1){
        for(int i=0; i<5; i++){
            Model_Func->getInputBufferFloat()[i] = input[i]/1023.0f;
        }
        Model_Func->predict();
        output = Model_Func->getOutputBufferFloat()[0];

        printf("x1: %f, x2: %f, x3: %f, x4: %f, x5: %f y: %f\n", input[0], input[1], input[2], input[3], input[4], output);
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}