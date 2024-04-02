#include "pch.h"
#include "StableDiffusionSessionParameters.h"
#include "MachineLearning/Executors/DmlExecutor.h"
#include "MachineLearning/Executors/CpuExecutor.h"

using namespace Axodox::MachineLearning::Executors;
using namespace Axodox::MachineLearning::Sessions;
using namespace Axodox::Storage;
using namespace std;

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  StableDiffusionSessionParameters::StableDiffusionSessionParameters(
    const std::shared_ptr<Sessions::OnnxEnvironment>& environment, 
    const std::shared_ptr<Executors::OnnxExecutor>& executor) :
    _environment(environment ? environment : make_unique<OnnxEnvironment>("Stable diffusion")),
    _gpuExecutor(executor ? executor : make_unique<DmlExecutor>()),
    _cpuExecutor(make_unique<CpuExecutor>())
  { }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::TextTokenizer() const
  {
    return { _environment, _cpuExecutor, OnnxModelSource::FromFilePath(lib_folder() / "custom_op_cliptok.onnx") };
  }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::TextEncoder() const
  {
    return { _environment, _gpuExecutor, ResolveModel("text_encoder") };
  }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::TextEncoder2() const
  {
    return { _environment, _gpuExecutor, ResolveModel("text_encoder_2") };
  }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::VaeEncoder() const
  {
    return { _environment, _gpuExecutor, ResolveModel("vae_encoder") };
  }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::ControlNet() const
  {
    return { _environment, _gpuExecutor, ResolveModel("controlnet") };
  }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::UNet() const
  {
    return { _environment, _gpuExecutor, ResolveModel("unet") };
  }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::VaeDecoder() const
  {
    return { _environment, _gpuExecutor, ResolveModel("vae_decoder") };
  }

  Sessions::OnnxSessionParameters StableDiffusionSessionParameters::SafetyChecker() const
  {
    return { _environment, _gpuExecutor, ResolveModel("safety_checker") };
  }

  StableDiffusionDirectorySessionParameters::StableDiffusionDirectorySessionParameters(
    const std::filesystem::path& directory,
    const std::shared_ptr<Sessions::OnnxEnvironment>& environment,
    const std::shared_ptr<Executors::OnnxExecutor>& executor) :
    StableDiffusionSessionParameters(environment, executor),
    _directory(directory)
  { }

  std::unique_ptr<Sessions::OnnxModelSource> StableDiffusionDirectorySessionParameters::ResolveModel(std::string_view type) const
  {
    return OnnxModelSource::FromFilePath(_directory / type / "model.onnx");
  }
}