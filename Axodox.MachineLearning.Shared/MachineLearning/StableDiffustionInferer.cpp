#include "pch.h"
#include "StableDiffustionInferer.h"
#include "VaeDecoder.h"
#include "OnnxModelMetadata.h"

using namespace Axodox::Infrastructure;
using namespace DirectX;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  StableDiffusionInferer::StableDiffusionInferer(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / L"unet/model.onnx")))
  {
    auto metadata = OnnxModelMetadata::Create(_environment, _session);
    _hasTextEmbeds = metadata.Inputs.contains("text_embeds");
    _hasTimeIds = metadata.Inputs.contains("time_ids");
    _isUsingFloat16 = metadata.Inputs["sample"].Type == TensorType::Half;
    _vaeScalingFactor = _hasTextEmbeds ? 0.13025f : 0.18215f;

    _session.Evict();
    _logger.log(log_severity::information, "Loaded.");
  }

  Tensor StableDiffusionInferer::RunInference(const StableDiffusionOptions& options, Threading::async_operation_source* async)
  {
    _logger.log(log_severity::information, "Running inference...");

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

    context.Scheduler = StableDiffusionScheduler::Create(options.SchedulerType, { .InferenceStepCount = options.StepCount, .Randoms = context.Randoms });

    //Schedule steps
    //list<Tensor> derivatives;
    //auto steps = context.Scheduler.GetSteps(options.StepCount);
    auto initialStep = size_t(clamp(int(options.StepCount - options.StepCount * options.DenoisingStrength - 1), 0, int(options.StepCount)));

    //Create initial sample
    auto latentSample = options.LatentInput ? PrepareLatentSample(context, options.LatentInput, context.Scheduler->Sigmas()[initialStep], _vaeScalingFactor) : GenerateLatentSample(context);

    //Bind constant inputs    
    IoBinding binding{ _session };
    binding.BindOutput("out_sample", _environment->MemoryInfo());

    auto embeddingCount = options.TextEmbeddings.Weights.size();
    if (holds_alternative<EncodedText>(options.TextEmbeddings.Tensor))
    {
      auto& encodedText = get<EncodedText>(options.TextEmbeddings.Tensor);
      binding.BindInput("encoder_hidden_states", encodedText.LastHiddenState.ToHalf(_isUsingFloat16).Duplicate(options.BatchSize).ToOrtValue());
      if (_hasTextEmbeds && encodedText.TextEmbeds) binding.BindInput("text_embeds", encodedText.LastHiddenState.ToHalf(_isUsingFloat16).Duplicate(options.BatchSize).ToOrtValue());
      if (_hasTimeIds) binding.BindInput("time_ids", GetTimeIds().ToHalf(_isUsingFloat16).Duplicate(encodedText.LastHiddenState.Shape[0] * options.BatchSize).ToOrtValue());
    }

    //Run iteration
    const EncodedText* currentEmbedding = nullptr;
    for (size_t i = initialStep; i < context.Scheduler->Timesteps().size(); i++)
    {
      auto timestep = context.Scheduler->Timesteps()[i];
      auto sigma = context.Scheduler->Sigmas()[i];

      _logger.log(log_severity::information, "Step {}/{}...", i + 1, context.Scheduler->Timesteps().size());

      //Update status
      if (async)
      {
        async->update_state((i + 1.f) / options.StepCount, format("Denoising {}/{}...", i + 1, options.StepCount));
        if (async->is_cancelled()) return {};
      }

      //Update embeddings
      if (holds_alternative<ScheduledTensor>(options.TextEmbeddings.Tensor))
      {
        auto embedding = get<ScheduledTensor>(options.TextEmbeddings.Tensor)[i].get();
        if (currentEmbedding != embedding)
        {
          currentEmbedding = embedding;
          binding.BindInput("encoder_hidden_states", currentEmbedding->LastHiddenState.ToHalf(_isUsingFloat16).Duplicate(options.BatchSize).ToOrtValue());
          if (_hasTextEmbeds && currentEmbedding->TextEmbeds) binding.BindInput("text_embeds", currentEmbedding->TextEmbeds.ToHalf(_isUsingFloat16).Duplicate(options.BatchSize).ToOrtValue());
          if (_hasTimeIds) binding.BindInput("time_ids", GetTimeIds().ToHalf(_isUsingFloat16).Duplicate(currentEmbedding->LastHiddenState.Shape[0] * options.BatchSize).ToOrtValue());
        }
      }

      //Update sample
      auto scaledSample = latentSample.Duplicate(embeddingCount).Swizzle(options.BatchSize) / sqrt(sigma * sigma + 1);
      binding.BindInput("sample", scaledSample.ToHalf(_isUsingFloat16).ToOrtValue());

      //Update timestep
      binding.BindInput("timestep", Tensor(timestep).ToHalf(_isUsingFloat16).ToOrtValue());

      //Run inference
      _session.Run({}, binding);

      //Read output
      auto outputs = binding.GetOutputValues();
      auto output = Tensor::FromOrtValue(outputs[0]).ToSingle();

      auto outputComponents = output.Swizzle(embeddingCount).Split(embeddingCount);

      //Calculate guidance
      Tensor guidedNoise;
      for (auto embeddingIndex = 0; embeddingIndex < embeddingCount; embeddingIndex++)
      {
        auto componentWeight = options.TextEmbeddings.Weights[embeddingIndex];
        auto finalWeight = componentWeight * (componentWeight > 0.f ? options.GuidanceScale : options.GuidanceScale - 1.f);

        if (embeddingIndex == 0)
        {
          guidedNoise = outputComponents[embeddingIndex] * finalWeight;
        }
        else
        {
          guidedNoise.UnaryOperation<float>(outputComponents[embeddingIndex], [=](float a, float b) { return a + b * finalWeight; });
        }
      }

      //Refine latent image
      latentSample = context.Scheduler->ApplyStep(latentSample, guidedNoise, i);

      //Apply mask
      if (options.MaskInput)
      {
        auto maskedSample = PrepareLatentSample(context, options.LatentInput, sigma, _vaeScalingFactor);
        latentSample = BlendLatentSamples(maskedSample, latentSample, options.MaskInput);
      }
    }

    //Decode sample
    latentSample = latentSample * (1.0f / _vaeScalingFactor);

    _logger.log(log_severity::information, "Inference finished.");

    _session.Evict();
    _logger.log(log_severity::information, "Session evicted.");

    return latentSample;
  }

  Tensor StableDiffusionInferer::PrepareLatentSample(StableDiffusionContext& context, const Tensor& latents, float initialSigma, float vaeScalingFactor)
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

    result.UnaryOperation<float>(replicatedLatents, [=](float a, float b) { return a * initialSigma + b * vaeScalingFactor; });

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
    TensorShape shape{ context.Options->BatchSize, 4, context.Options->Height / 8, context.Options->Width / 8 };
    return Tensor::CreateRandom(shape, context.Randoms, context.Scheduler->Sigmas()[0]);
  }

  ImageDiffusionInfererKind StableDiffusionInferer::Type() const
  {
    return ImageDiffusionInfererKind::StableDiffusion;
  }

  void StableDiffusionOptions::Validate() const
  {
    if (MaskInput && !LatentInput) throw logic_error("Mask input cannot be set without latent input!");
    if (MaskInput && (MaskInput.Shape[2] != LatentInput.Shape[2] || MaskInput.Shape[3] != LatentInput.Shape[3])) throw logic_error("Mask and latent inputs must have a matching width and height.");
    if (holds_alternative<ScheduledTensor>(TextEmbeddings.Tensor) && get<ScheduledTensor>(TextEmbeddings.Tensor).size() != StepCount) throw logic_error("Scheduled text embedding size must match sample count.");

    auto embeddingDimension = (holds_alternative<EncodedText>(TextEmbeddings.Tensor) ? get<EncodedText>(TextEmbeddings.Tensor) : *get<ScheduledTensor>(TextEmbeddings.Tensor)[0]).LastHiddenState.Shape[0];
    if (TextEmbeddings.Weights.size() != embeddingDimension) throw logic_error("Scheduled text embedding weight count does not match the first dimension of the text embedding tensor.");
  }

  Tensor StableDiffusionInferer::GetTimeIds() const
  {
    Tensor result{ TensorType::Single, 1, 6, 0, 0 };

    auto values = result.AsSpan<float>();

    //Original size
    values[0] = 1024.f;
    values[1] = 1024.f;

    //Crop coords
    values[2] = 0.f;
    values[3] = 0.f;

    //Target size
    values[4] = 1024.f;
    values[5] = 1024.f;

    return result;
  }
}