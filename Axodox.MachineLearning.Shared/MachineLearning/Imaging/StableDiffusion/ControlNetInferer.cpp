#include "pch.h"
#include "ControlNetInferer.h"

using namespace Axodox::MachineLearning::Imaging::StableDiffusion::Schedulers;
using namespace DirectX;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  ControlNetInferer::ControlNetInferer(const Sessions::OnnxSessionParameters& controlnetParameters, const Sessions::OnnxSessionParameters& unetParameters) :
    _controlnetSessionContainer(controlnetParameters),
    _unetSessionContainer(unetParameters)
  { }

  ControlNetInferer::ControlNetInferer(const Sessions::OnnxSessionParameters& controlnetParameters, const StableDiffusionDirectorySessionParameters& unetParameters) :
    ControlNetInferer(controlnetParameters, unetParameters.ControlNet())
  { }

  Tensor ControlNetInferer::RunInference(const ControlNetOptions& options, Threading::async_operation_source* async)
  {
    //Validate inputs
    options.Validate();

    //Build context
    if (async) async->update_state("Preparing latent sample...");
    
    StableDiffusionContext context{
      .Options = &options
    };

    context.Randoms.reserve(options.BatchSize);
    for (size_t i = 0; i < options.BatchSize; i++)
    {
      context.Randoms.push_back(minstd_rand{ options.Seed + uint32_t(i) });
    }

    context.Scheduler = StableDiffusionScheduler::Create(options.Scheduler, { .InferenceStepCount = options.StepCount, .Randoms = context.Randoms });

    //Schedule steps
    auto initialStep = size_t(clamp(int(options.StepCount - options.StepCount * options.DenoisingStrength - 1), 0, int(options.StepCount)));

    //Create initial sample
    auto latentSample = options.LatentInput ? StableDiffusionInferer::PrepareLatentSample(context, options.LatentInput, context.Scheduler->Sigmas()[initialStep]) : StableDiffusionInferer::GenerateLatentSample(context);

    //Get session
    auto controlnetEnvironment = _controlnetSessionContainer.Environment();
    auto controlnetSession = _controlnetSessionContainer.Session();

    auto unetEnvironment = _unetSessionContainer.Environment();
    auto unetSession = _unetSessionContainer.Session();

    //Bind constant inputs / outputs 
    IoBinding controlnetBinding{ *controlnetSession };
    for (auto i = 0; i < 12; i++)
    {
      controlnetBinding.BindOutput(format("down_block_{}_additional_residual", i).c_str(), controlnetEnvironment->MemoryInfo());
    }
    controlnetBinding.BindOutput("mid_block_additional_residual", controlnetEnvironment->MemoryInfo());
    controlnetBinding.BindInput("controlnet_cond", options.ConditionInput.ToHalf().ToOrtValue());
    controlnetBinding.BindInput("conditioning_scale", Tensor(double(options.ConditioningScale)).ToOrtValue());
    
    IoBinding unetBinding{ *unetSession };
    unetBinding.BindOutput("out_sample", unetEnvironment->MemoryInfo());

    auto embeddingCount = options.TextEmbeddings.Weights.size();
    if (holds_alternative<EncodedText>(options.TextEmbeddings.Tensor))
    {
      auto encoderHiddenStates = get<EncodedText>(options.TextEmbeddings.Tensor).LastHiddenState.ToHalf().Duplicate(options.BatchSize);
      controlnetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());
      unetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());
    }

    //Run iteration
    const EncodedText* currentEmbedding = nullptr;
    for (size_t i = initialStep; i < context.Scheduler->Timesteps().size(); i++)
    {
      auto timestep = context.Scheduler->Timesteps()[i];
      auto sigma = context.Scheduler->Sigmas()[i];

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
          auto encoderHiddenStates = embedding->LastHiddenState.ToHalf().Duplicate(options.BatchSize);
          controlnetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());
          unetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());

          currentEmbedding = embedding;
        }
      }

      //Update sample
      {
        auto scaledSample = (latentSample.Duplicate(embeddingCount).Swizzle(options.BatchSize) / sqrt(sigma * sigma + 1)).ToHalf();
        controlnetBinding.BindInput("sample", scaledSample.ToOrtValue());
        unetBinding.BindInput("sample", scaledSample.ToOrtValue());
      }

      //Update timestep
      {
        auto timestepTensor = Tensor(timestep).ToHalf();
        controlnetBinding.BindInput("timestep", timestepTensor.ToOrtValue());
        unetBinding.BindInput("timestep", timestepTensor.ToOrtValue());
      }

      //Run ControlNet
      controlnetSession->Run({}, controlnetBinding);

      //Read outputs and feed them into UNet
      {
        auto controlnetOutputs = controlnetBinding.GetOutputValues();
        for (auto i = 0; i < 12; i++)
        {
          unetBinding.BindInput(format("down_block_{}_additional_residual", i).c_str(), controlnetOutputs[i]);
        }
        unetBinding.BindInput("mid_block_additional_residual", controlnetOutputs.back());
      }

      //Run UNet
      unetSession->Run({}, unetBinding);

      //Read output
      auto outputs = unetBinding.GetOutputValues();
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
        auto maskedSample = StableDiffusionInferer::PrepareLatentSample(context, options.LatentInput, sigma);
        latentSample = StableDiffusionInferer::BlendLatentSamples(maskedSample, latentSample, options.MaskInput);
      }
    }

    //Decode sample
    latentSample = latentSample * (1.0f / 0.18215f);
    return latentSample;
  }

  ImageDiffusionInfererKind ControlNetInferer::Type() const
  {
    return ImageDiffusionInfererKind::ControlNet;
  }

  void ControlNetOptions::Validate() const
  {
    StableDiffusionOptions::Validate();
    //TBD
  }
}