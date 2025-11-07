#ifndef __NNModel__
#define __NNModel__

#include <stdint.h>
#define NOPS 2

namespace tflite{
    template <unsigned int tOpCount> class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

class NNModel{
private:
    tflite::MicroMutableOpResolver<NOPS> *resolver;
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *nnModel;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    TfLiteTensor *hidden;
    uint8_t *tensor_arena;

public:
    NNModel(int kArenaSize, const unsigned char *model);
    int8_t* getInputBufferInt8();
    uint8_t* getInputBufferUInt8();
    int32_t* getInputBufferInt32();
    uint32_t* getInputBufferUInt32();
    float *getInputBufferFloat();
    float getInputScale();
    int getInputZeroPoint();
    void predict();
    int8_t* getOutputBufferInt8();
    uint8_t* getOutputBufferUInt8();
    int32_t* getOutputBufferInt32();
    uint32_t* getOutputBufferUInt32();
    float* getOutputBufferFloat();
    int getOutputDims();
    float getOutputScale();
    int getOutputZeroPoint();
};

#endif