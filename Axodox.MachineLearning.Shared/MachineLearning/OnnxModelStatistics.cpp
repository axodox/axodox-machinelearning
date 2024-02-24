#include "pch.h"
#include "OnnxModelStatistics.h"
#include "Infrastructure/Logger.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  string_view TensorInfoToString(ONNXTensorElementDataType type)
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

  void OnnxPrintPropertyInfos(Session& session, Allocator& allocator,
    size_t(detail::ConstSessionImpl<OrtSession>::* getCount)() const,
    AllocatedStringPtr(detail::ConstSessionImpl<OrtSession>::* getNameAllocated)(size_t, OrtAllocator*) const,
    TypeInfo(detail::ConstSessionImpl<OrtSession>::* getTypeInfo)(size_t) const)
  {
    Infrastructure::logger logger{ "OnnxStatistics" };

    auto count = (session.*getCount)();
    for (size_t i = 0; i < count; i++)
    {
      auto name = (session.*getNameAllocated)(i, allocator);
      auto typeInfo = (session.*getTypeInfo)(i);
      auto typeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
      auto type = TensorInfoToString(typeAndShapeInfo.GetElementType());
      
      vector<const char*> dimensionNames;
      dimensionNames.resize(typeAndShapeInfo.GetDimensionsCount());
      typeAndShapeInfo.GetSymbolicDimensions(dimensionNames.data(), dimensionNames.size());

      auto shape = OnnxTensorShapeToString(typeAndShapeInfo.GetShape(), dimensionNames);
      logger.log(log_severity::information, "    {}: {} ({})", name.get(), type.data(), shape.c_str());
    }
  }

  void OnnxPrintStatistics(OnnxEnvironment& environment, Ort::Session& session)
  {
    Infrastructure::logger logger{ "OnnxStatistics" };

    Allocator allocator{ session, environment->MemoryInfo() };
    logger.log(log_severity::information, "Graph statistics");

    //Name
    auto graphName = session.GetModelMetadata().GetGraphNameAllocated(allocator);
    logger.log(log_severity::information, "  Name: {}", graphName.get());

    //Initializers
    logger.log(log_severity::information, "  Initializers:");
    OnnxPrintPropertyInfos(session, allocator, &Session::GetOverridableInitializerCount, &Session::GetOverridableInitializerNameAllocated, &Session::GetOverridableInitializerTypeInfo);

    //Inputs
    logger.log(log_severity::information, "  Inputs:");
    OnnxPrintPropertyInfos(session, allocator, &Session::GetInputCount, &Session::GetInputNameAllocated, &Session::GetInputTypeInfo);

    //Outputs
    logger.log(log_severity::information, "  Outputs:");
    OnnxPrintPropertyInfos(session, allocator, &Session::GetOutputCount, &Session::GetOutputNameAllocated, &Session::GetOutputTypeInfo);
  }
}