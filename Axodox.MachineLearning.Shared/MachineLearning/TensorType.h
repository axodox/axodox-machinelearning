#pragma once
#include "pch.h"
#include "Infrastructure/Half.h"

namespace Axodox::MachineLearning
{
  enum class TensorType
  {
    Unknown,
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Half,
    Single,
    Double
  };

  template<typename T>
  constexpr TensorType ToTensorType()
  {
    if constexpr (std::is_same_v<T, bool>)
    {
      return TensorType::Bool;
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
      return TensorType::UInt8;
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
      return TensorType::UInt16;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
      return TensorType::UInt32;
    }
    else if constexpr (std::is_same_v<T, uint64_t>)
    {
      return TensorType::UInt64;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
      return TensorType::Int8;
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
      return TensorType::Int16;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
      return TensorType::Int32;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
      return TensorType::Int64;
    }
    else if constexpr (std::is_same_v<T, Infrastructure::half>)
    {
      return TensorType::Half;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
      return TensorType::Single;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
      return TensorType::Double;
    }
    else
    {
      return TensorType::Unknown;
    }
  }

  AXODOX_MACHINELEARNING_API size_t GetElementSize(TensorType type);

#ifdef USE_ONNX
  AXODOX_MACHINELEARNING_API TensorType ToTensorType(ONNXTensorElementDataType type);
  AXODOX_MACHINELEARNING_API ONNXTensorElementDataType ToTensorType(TensorType type);
#endif
}