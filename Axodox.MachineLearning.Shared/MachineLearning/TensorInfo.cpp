#include "pch.h"
#include "TensorInfo.h"
#include "Infrastructure/BitwiseOperations.h"

using namespace Axodox::Infrastructure;
using namespace std;

namespace Axodox::MachineLearning
{
  TensorInfo TensorInfo::FromTypeAndShapeInfo(const Ort::TensorTypeAndShapeInfo& info)
  {
    return FromTypeAndShapeInfo(info.GetConst());
  }

  TensorInfo TensorInfo::FromTypeAndShapeInfo(const Ort::ConstTensorTypeAndShapeInfo& info)
  {
    TensorInfo result;
    zero_memory(result);

    //Convert type
    result.Type = ToTensorType(info.GetElementType());

    //Convert shape
    auto shape = info.GetShape();
    for (auto i = 0; auto dimension : shape)
    {
      if (dimension > 0)
      {
        result.Shape[i++] = size_t(dimension);
      }
    }

    //Convert dimensions
    auto dimensionCount = info.GetDimensionsCount();
    info.GetSymbolicDimensions(result.Dimensions.data(), min(result.Dimensions.size(), dimensionCount));

    return result;
  }
}