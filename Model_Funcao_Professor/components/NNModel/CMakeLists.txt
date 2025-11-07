idf_component_register(
    SRCS "NNModel.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro esp-nn freertos
)