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

    void Validate() const;
  };

  struct StableDiffusionContext
  {
    const StableDiffusionOptions* Options;
    StableDiffusionScheduler Scheduler;
    std::vector<std::minstd_rand> Randoms;
  };

  enum class ImageDiffusionInfererKind
  {
    StableDiffusion,
    ControlNet
  };

  class AXODOX_MACHINELEARNING_API ImageDiffusionInferer
  {
  public:
    virtual ~ImageDiffusionInferer() = default;

    virtual ImageDiffusionInfererKind Type() const = 0;
  };

  class AXODOX_MACHINELEARNING_API StableDiffusionInferer : public ImageDiffusionInferer
  {
  public:
    StableDiffusionInferer(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor RunInference(const StableDiffusionOptions& options, Threading::async_operation_source* async = nullptr);

    static Tensor GenerateLatentSample(StableDiffusionContext& context);
    static Tensor PrepareLatentSample(StableDiffusionContext& context, const Tensor& latents, float initialSigma);
    static Tensor BlendLatentSamples(const Tensor& a, const Tensor& b, const Tensor& weights);

    virtual ImageDiffusionInfererKind Type() const override;

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}