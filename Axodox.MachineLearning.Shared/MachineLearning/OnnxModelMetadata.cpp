#include "pch.h"
#include "OnnxModelMetadata.h"

using namespace Axodox::MachineLearning;
using namespace Ort;
using namespace std;

namespace {
  std::unordered_map<std::string, TensorInfo> GetProperties(
    Session& session,
    Allocator& allocator,
    size_t(detail::ConstSessionImpl<OrtSession>::* getCount)() const,
    AllocatedStringPtr(detail::ConstSessionImpl<OrtSession>::* getNameAllocated)(size_t, OrtAllocator*) const,
    TypeInfo(detail::ConstSessionImpl<OrtSession>::* getTypeInfo)(size_t) const)
  {
    unordered_map<string, TensorInfo> results;

    auto count = (session.*getCount)();
    for (size_t i = 0; i < count; i++)
    {
      auto name = string((session.*getNameAllocated)(i, allocator).get());
      auto info = TensorInfo::FromTypeAndShapeInfo(ConstTensorTypeAndShapeInfo((session.*getTypeInfo)(i).GetTensorTypeAndShapeInfo()));
      results[name] = info;
    }

    return results;
  }
}

namespace Axodox::MachineLearning
{
  OnnxModelMetadata OnnxModelMetadata::Create(OnnxEnvironment& environment, Ort::Session& session)
  {
    Allocator allocator{ session, environment->MemoryInfo() };

    OnnxModelMetadata result;
    result.Name = string(session.GetModelMetadata().GetGraphNameAllocated(allocator).get());
    result.Initializers = GetProperties(session, allocator, &Session::GetOverridableInitializerCount, &Session::GetOverridableInitializerNameAllocated, &Session::GetOverridableInitializerTypeInfo);
    result.Inputs = GetProperties(session, allocator, &Session::GetInputCount, &Session::GetInputNameAllocated, &Session::GetInputTypeInfo);
    result.Outputs = GetProperties(session, allocator, &Session::GetOutputCount, &Session::GetOutputNameAllocated, &Session::GetOutputTypeInfo);

    return result;
  }
}