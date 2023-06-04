#include "pch.h"
#include "OnnxModelStatistics.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  string_view OnnxTensorTypeToString(ONNXTensorElementDataType type)
  {
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      return "undefined";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "complex64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "complex128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    default:
      throw logic_error("Unsupported tensor type.");
    }
  }

  string OnnxTensorShapeToString(span<const int64_t> shape, span<const char*> names)
  {
    stringstream stream;
    for (size_t i = 0; auto size : shape)
    {
      if (stream.tellp() != 0) stream << ", ";

      if (size < 0)
      {
        stream << (names[i] ? names[i] : "-1");
      }
      else
      {
        stream << size;
      }

      i++;
    }
    return stream.str();
  }

  void OnnxPrintStatistics(OnnxEnvironment& environment, Ort::Session& session)
  {
    Allocator allocator{ session, environment.MemoryInfo() };
    printf("Graph statistics\n");

    auto graphName = session.GetModelMetadata().GetGraphNameAllocated(allocator);
    printf("  Name: %s\n", graphName.get());

    auto inputCount = session.GetInputCount();
    printf("  Inputs: %zd\n", inputCount);

    for (size_t i = 0; i < inputCount; i++)
    {
      auto inputName = session.GetInputNameAllocated(i, allocator);
      auto typeAndShapeInfo = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto inputType = OnnxTensorTypeToString(typeAndShapeInfo.GetElementType());

      vector<const char*> inputDimensionNames;
      inputDimensionNames.resize(typeAndShapeInfo.GetDimensionsCount());
      typeAndShapeInfo.GetSymbolicDimensions(inputDimensionNames.data(), inputDimensionNames.size());

      auto inputShape = OnnxTensorShapeToString(typeAndShapeInfo.GetShape(), inputDimensionNames);            
      printf("    %s: %s {%s}\n", inputName.get(), inputType.data(), inputShape.c_str());
    }

    auto outputCount = session.GetOutputCount();
    printf("  Outputs: %zd\n", outputCount);

    for (size_t i = 0; i < outputCount; i++)
    {
      auto outputName = session.GetOutputNameAllocated(i, allocator);
      auto typeAndShapeInfo = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto outputType = OnnxTensorTypeToString(typeAndShapeInfo.GetElementType());

      vector<const char*> outputDimensionNames;
      outputDimensionNames.resize(typeAndShapeInfo.GetDimensionsCount());
      typeAndShapeInfo.GetSymbolicDimensions(outputDimensionNames.data(), outputDimensionNames.size());

      auto outputShape = OnnxTensorShapeToString(typeAndShapeInfo.GetShape(), outputDimensionNames);
      printf("    %s: %s {%s}\n", outputName.get(), outputType.data(), outputShape.c_str());
    }
  }
}