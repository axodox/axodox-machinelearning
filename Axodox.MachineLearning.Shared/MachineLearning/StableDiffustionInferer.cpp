#include "pch.h"
#include "StableDiffustionInferer.h"
#include "VaeDecoder.h"
#include "OnnxModelStatistics.h"

using namespace DirectX;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  StableDiffusionInferer::StableDiffusionInferer(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment.CreateSession(source ? *source : (_environment.RootPath() / L"unet/model.onnx")))
  { }

  Tensor StableDiffusionInferer::RunInference(const StableDiffusionOptions& options, Threading::async_operation_source* async)
  {
    //Validate inputs
    options.Validate();

    if (async) async->update_state("Preparing latent sample...");

    //Build context
    StableDiffusionContext context{
      .Options = &options
    };

    context.Randoms.reserve(options.BatchSize);
    for (size_t i = 0; i < options.BatchSize; i++)
    {
      context.Randoms.push_back(minstd_rand{ options.Seed + uint32_t(i) });
    }

    //Schedule steps
    list<Tensor> derivatives;
    auto steps = context.Scheduler.GetSteps(options.StepCount);
    auto initialStep = size_t(clamp(int(options.StepCount - options.StepCount * options.DenoisingStrength - 1), 0, int(options.StepCount)));

    //Create initial sample
    auto latentSample = options.LatentInput ? PrepareLatentSample(context, options.LatentInput, steps.Sigmas[initialStep]) : GenerateLatentSample(context);

    //Bind constant inputs    
    IoBinding binding{ _session };
    binding.BindOutput("out_sample", _environment.MemoryInfo());

    if (holds_alternative<Tensor>(options.TextEmbeddings))
    {
      binding.BindInput("encoder_hidden_states", get<Tensor>(options.TextEmbeddings).ToHalf().Duplicate(options.BatchSize).ToOrtValue());
    }

    //Run iteration
    const Tensor* currentEmbedding = nullptr;
    for (size_t i = initialStep; i < steps.Timesteps.size(); i++)
    {
      //Update status
      if (async)
      {
        async->update_state((i + 1.f) / options.StepCount, format("Denoising {}/{}...", i + 1, options.StepCount));
        if (async->is_cancelled()) return {};
      }

      //Update embeddings
      if (holds_alternative<ScheduledTensor>(options.TextEmbeddings))
      {
        auto embedding = get<ScheduledTensor>(options.TextEmbeddings)[i].get();
        if (currentEmbedding != embedding)
        {
          currentEmbedding = embedding;
          binding.BindInput("encoder_hidden_states", currentEmbedding->ToHalf().Duplicate(options.BatchSize).ToOrtValue());
        }
      }

      //Update sample
      auto scaledSample = latentSample.Duplicate().Swizzle(options.BatchSize) / sqrt(steps.Sigmas[i] * steps.Sigmas[i] + 1);
      binding.BindInput("sample", scaledSample.ToHalf().ToOrtValue());

      //Update timestep
      binding.BindInput("timestep", Tensor(steps.Timesteps[i]).ToHalf().ToOrtValue());

      //Run inference
      _session.Run({}, binding);

      //Read output
      auto outputs = binding.GetOutputValues();
      auto output = Tensor::FromOrtValue(outputs[0]).ToSingle();

      auto outputComponents = output.Swizzle().Split();

      //Calculate guidance
      auto& blankNoise = outputComponents[0];
      auto& textNoise = outputComponents[1];
      auto guidedNoise = blankNoise.BinaryOperation<float>(textNoise, [guidanceScale = options.GuidanceScale](float a, float b)
        { return a + guidanceScale * (b - a); });

      //Refine latent image
      latentSample = steps.ApplyStep(latentSample, guidedNoise, derivatives, context.Randoms, i);

      //Apply mask
      if (options.MaskInput)
      {
        auto maskedSample = PrepareLatentSample(context, options.LatentInput, steps.Sigmas[i]);
        latentSample = BlendLatentSamples(maskedSample, latentSample, options.MaskInput);
      }
    }

    //Decode sample
    latentSample = latentSample * (1.0f / 0.18215f);
    return latentSample;
  }

  Tensor StableDiffusionInferer::PrepareLatentSample(StableDiffusionContext& context, const Tensor& latents, float initialSigma)
  {
    auto replicatedLatents = latents.DuplicateToSize(context.Options->BatchSize);

    auto result = Tensor::CreateRandom(replicatedLatents.Shape, context.Randoms);
    
    if (context.Options->MaskInput)
    {
      auto maskInput = context.Options->MaskInput.Duplicate(latents.Shape[1]);
      swap(maskInput.Shape[0], maskInput.Shape[1]);

      result.UnaryOperation<float>(maskInput, [=](float a, float b) { return a * b; });
      //replicatedLatents.UnaryOperation<float>(maskInput, [=](float a, float b) { return a * (1.f - floor(b)); });
    }

    result.UnaryOperation<float>(replicatedLatents, [=](float a, float b) { return a * initialSigma + b * 0.18215f; });

    return result;
  }

  Tensor StableDiffusionInferer::BlendLatentSamples(const Tensor& a, const Tensor& b, const Tensor& weights)
  {
    if (a.Shape != b.Shape) throw logic_error("Tensor sizes must match!");
    if (weights.Shape[0] != 1 || weights.Shape[1] != 1 || weights.Shape[2] != a.Shape[2] || weights.Shape[3] != a.Shape[3]) throw logic_error("Weight tensor mismatches the size of the blended tensors!");

    Tensor result{ TensorType::Single, a.Shape };

    for (size_t i = 0; i < a.Shape[0]; i++)
    {
      for (size_t j = 0; j < a.Shape[1]; j++)
      {
        auto pA = a.AsPointer<float>(i, j);
        auto pB = b.AsPointer<float>(i, j);
        auto pC = result.AsPointer<float>(i, j);
        for (auto weight : weights.AsSpan<float>())
        {
          *pC++ = lerp(*pA++, *pB++, weight);
        }
      }
    }

    return result;
  }

  Tensor StableDiffusionInferer::GenerateLatentSample(StableDiffusionContext& context)
  {
    Tensor::shape_t shape{ context.Options->BatchSize, 4, context.Options->Height / 8, context.Options->Width / 8 };
    return Tensor::CreateRandom(shape, context.Randoms, context.Scheduler.InitialNoiseSigma());
  }

  ImageDiffusionInfererKind StableDiffusionInferer::Type() const
  {
    return ImageDiffusionInfererKind::StableDiffusion;
  }

  void StableDiffusionOptions::Validate() const
  {
    if (MaskInput && !LatentInput) throw logic_error("Mask input cannot be set without latent input!");
    if (MaskInput && (MaskInput.Shape[2] != LatentInput.Shape[2] || MaskInput.Shape[3] != LatentInput.Shape[3])) throw logic_error("Mask and latent inputs must have a matching width and height.");
    if (holds_alternative<ScheduledTensor>(TextEmbeddings) && get<ScheduledTensor>(TextEmbeddings).size() != StepCount) throw logic_error("Scheduled text embedding size must match sample count.");
  }
}