#include "NNModel.h"
#include "esp_heap_caps.h"
//#include "esp_psram.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

NNModel::NNModel(int kArenaSize, const unsigned char *model){
    nnModel = tflite::GetModel(model);
    if (nnModel->version() != TFLITE_SCHEMA_VERSION)    {
        MicroPrintf("Model provided is schema version %d not equal to supported version %d.",
                    nnModel->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<NOPS>();
    // Add Ops here:
    resolver->AddFullyConnected();
    resolver->AddLeakyRelu();

    /*if (esp_psram_is_initialized()) {
        MicroPrintf("PSRAM detected → using heap_caps_malloc()");
        tensor_arena = (uint8_t *)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    } else {*/
        MicroPrintf("No PSRAM → using malloc()");
        tensor_arena = (uint8_t *)malloc(kArenaSize);
    //}

    if (!tensor_arena)    {
        MicroPrintf("Could not allocate arena");
        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        nnModel, *resolver, tensor_arena, kArenaSize);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)    {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    MicroPrintf("Used bytes %d\n", used_bytes);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

int8_t *NNModel::getInputBufferInt8(){
    return input->data.int8;
}

uint8_t *NNModel::getInputBufferUInt8(){
    return input->data.uint8;
}

int32_t *NNModel::getInputBufferInt32(){
    return input->data.i32;
}

uint32_t *NNModel::getInputBufferUInt32(){
    return input->data.u32;
}

float *NNModel::getInputBufferFloat(){
    return input->data.f;
}

float NNModel::getInputScale(){
    return input->params.scale;
}

int NNModel::getInputZeroPoint(){
    return input->params.zero_point;
}

void NNModel::predict(){
    interpreter->Invoke();
}

int8_t *NNModel::getOutputBufferInt8(){
    return output->data.int8;
}

uint8_t *NNModel::getOutputBufferUInt8(){
    return output->data.uint8;
}

int32_t *NNModel::getOutputBufferInt32(){
    return output->data.i32;
}

uint32_t *NNModel::getOutputBufferUInt32(){
    return output->data.u32;
}

float *NNModel::getOutputBufferFloat(){
    return output->data.f;
}

int NNModel::getOutputDims(){
    return output->dims->data[1];
}

float NNModel::getOutputScale(){
    return output->params.scale;
}

int NNModel::getOutputZeroPoint(){
    return output->params.zero_point;
}