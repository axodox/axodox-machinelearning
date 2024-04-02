#include "pch.h"
#include "CpuExecutor.h"

using namespace Axodox::Storage;

namespace Axodox::MachineLearning::Executors
{
  void CpuExecutor::Apply(Ort::SessionOptions& sessionOptions)
  {
    auto ortExtensionsPath = lib_folder() / L"ortextensions.dll";
    sessionOptions.RegisterCustomOpsLibrary(ortExtensionsPath.c_str());
  }

  void CpuExecutor::Ensure()
  { }
}