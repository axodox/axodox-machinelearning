#pragma once
#include "TextEncoder.h"
#include "OnnxEnvironment.h"
#include "StableDiffusionScheduler.h"
#include "Threading/AsyncOperation.h"

namespace Axodox::MachineLearning
{
  typedef std::vector<std::shared_ptr<EncodedText>> ScheduledTensor;

  struct TextEmbedding
  {
    std::variant<EncodedText, ScheduledTensor> Tensor;
    std::vector<float> Weights;
  };

  struct AXODOX_MACHINELEARNING_API StableDiffusionOptions
  {
    size_t StepCount = 15;    
    size_t BatchSize = 1;
    size_t Width = 512;
    size_t Height = 512;
    float GuidanceScale = 7.f;    
    uint32_t Seed = 0;
    TextEmbedding TextEmbeddings;
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
    static inline const Infrastructure::logger _logger{ "StableDiffusionInferer" };

  public:
    StableDiffusionInferer(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor RunInference(const StableDiffusionOptions& options, Threading::async_operation_source* async = nullptr);

    static Tensor GenerateLatentSample(StableDiffusionContext& context);
    static Tensor PrepareLatentSample(StableDiffusionContext& context, const Tensor& latents, float initialSigma, float vaeScalingFactor = 0.18215f);
    static Tensor BlendLatentSamples(const Tensor& a, const Tensor& b, const Tensor& weights);

    virtual ImageDiffusionInfererKind Type() const override;

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
    bool _hasTextEmbeds;
    bool _hasTimeIds;
    bool _isUsingFloat16;
    float _vaeScalingFactor;

    Tensor GetTimeIds() const;
  };
}