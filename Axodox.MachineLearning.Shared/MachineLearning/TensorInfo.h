#pragma once
#include "TensorType.h"

namespace Axodox::MachineLearning
{
  struct AXODOX_MACHINELEARNING_API TensorInfo
  {
    static const size_t TensorDimension = 4;
    typedef std::array<size_t, TensorDimension> TensorShape;

    TensorType Type;
    TensorShape Shape;

    static TensorInfo FromTypeAndShapeInfo(const Ort::TensorTypeAndShapeInfo& info);
    static TensorInfo FromTypeAndShapeInfo(const Ort::ConstTensorTypeAndShapeInfo& info);
  };
}