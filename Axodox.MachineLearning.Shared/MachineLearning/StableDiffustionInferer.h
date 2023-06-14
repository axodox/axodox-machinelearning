#pragma once
#include "Tensor.h"
#include "OnnxEnvironment.h"
#include "StableDiffusionScheduler.h"
#include "Threading/AsyncOperation.h"

namespace Axodox::MachineLearning
{
  typedef std::vector<std::shared_ptr<Tensor>> ScheduledTensor;

  struct AXODOX_MACHINELEARNING_API StableDiffusionOptions
  {
    size_t StepCount = 15;    
    size_t BatchSize = 1;
    size_t Width = 512;
    size_t Height = 512;
    float GuidanceScale = 7.f;    
    uint32_t Seed = 0;
    std::variant<Tensor, ScheduledTensor> TextEmbeddings;
    Tensor LatentInput;
    Tensor MaskInput;
    float DenoisingStrength = 1.f;
  };

  class AXODOX_MACHINELEARNING_API StableDiffusionInferer
  {
    struct StableDiffusionContext
    {
      StableDiffusionOptions Options;
      StableDiffusionScheduler Scheduler;
      std::vector<std::minstd_rand> Randoms;
    };

  public:
    StableDiffusionInferer(OnnxEnvironment& environment);

    Tensor RunInference(const StableDiffusionOptions& options, Threading::async_operation_source* async = nullptr);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
    bool _isHalfModel;

    static Tensor GenerateLatentSample(StableDiffusionContext& context);
    static Tensor PrepareLatentSample(StableDiffusionContext& context, const Tensor& latents, float initialSigma);
    static Tensor BlendLatentSamples(const Tensor& a, const Tensor& b, const Tensor& weights);
  };
}