#ifndef TYPES_H
#define TYPES_H

namespace types {

enum class LossFunction { MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy };
enum class ActivationFunction { ReLU, Tanh, Softmax, Sigmoid };
enum class TaskType { Classification, Regression };
enum class LayerType { Hidden, Output };
    
}

#endif // TYPES_H