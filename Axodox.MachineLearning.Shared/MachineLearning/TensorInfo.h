#pragma once
#include "TensorType.h"

namespace Axodox::MachineLearning
{
  static const size_t TensorDimension = 4;
  typedef std::array<size_t, TensorDimension> TensorShape;
  typedef std::array<const char*, TensorDimension> TensorDimensions;

  struct AXODOX_MACHINELEARNING_API TensorInfo
  {
    TensorType Type;
    TensorShape Shape;
    TensorDimensions Dimensions;

    static TensorInfo FromTypeAndShapeInfo(const Ort::TensorTypeAndShapeInfo& info);
    static TensorInfo FromTypeAndShapeInfo(const Ort::ConstTensorTypeAndShapeInfo& info);
  };
}