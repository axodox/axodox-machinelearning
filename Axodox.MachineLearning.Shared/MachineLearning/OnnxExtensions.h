#pragma once
#include "..\includes.h"

namespace Axodox::MachineLearning
{
  template<typename T>
  std::span<const T> AsSpan(Ort::Value& value)
  {
    auto data = value.GetTensorData<T>();
    auto shape = value.GetTensorTypeAndShapeInfo().GetShape();

    size_t size = 1;
    for (auto dimension : shape)
    {
      if(dimension > 0) size *= dimension;
    }

    return { data, size };
  }
}