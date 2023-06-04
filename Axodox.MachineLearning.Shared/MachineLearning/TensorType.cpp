#include "pch.h"
#include "TensorType.h"

using namespace std;

namespace Axodox::MachineLearning
{
  TensorType ToTensorType(ONNXTensorElementDataType type)
  {
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return TensorType::Bool;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return TensorType::UInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return TensorType::UInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return TensorType::UInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return TensorType::UInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return TensorType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return TensorType::Int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return TensorType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return TensorType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return TensorType::Half;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return TensorType::Single;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return TensorType::Double;
    default:
      return TensorType::Unknown;
    }
  }

  ONNXTensorElementDataType ToTensorType(TensorType type)
  {
    switch (type)
    {
    case TensorType::Unknown:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    case TensorType::Bool:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case TensorType::UInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case TensorType::UInt16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case TensorType::UInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case TensorType::UInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case TensorType::Int8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case TensorType::Int16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case TensorType::Int32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case TensorType::Int64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TensorType::Half:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TensorType::Single:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TensorType::Double:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    default:
      throw logic_error("Tensor type not implemented.");
    }
  }

  size_t GetElementSize(TensorType type)
  {
    switch (type)
    {
    case TensorType::Unknown:
      return 0;
    case TensorType::Bool:
      return 1;
    case TensorType::UInt8:
      return 1;
    case TensorType::UInt16:
      return 2;
    case TensorType::UInt32:
      return 4;
    case TensorType::UInt64:
      return 8;
    case TensorType::Int8:
      return 1;
    case TensorType::Int16:
      return 2;
    case TensorType::Int32:
      return 4;
    case TensorType::Int64:
      return 8;
    case TensorType::Half:
      return 2;
    case TensorType::Single:
      return 4;
    case TensorType::Double:
      return 8;
    default:
      throw logic_error("Tensor type not implemented.");
    }
  }
}