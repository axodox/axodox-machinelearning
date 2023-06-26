#include "pch.h"
#include "ControlNetInferer.h"

using namespace DirectX;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  ControlNetInferer::ControlNetInferer(OnnxEnvironment& environment, const std::filesystem::path& controlnetPath, std::optional<ModelSource> unetSource) :
    _environment(environment),
    _unetSession(environment.CreateSession(unetSource ? *unetSource : (_environment.RootPath() / L"controlnet/model.onnx"))),
    _controlnetPath(controlnetPath),
    _controlnetSession(nullptr)
  { }

  Tensor ControlNetInferer::RunInference(const ControlNetOptions& options, Threading::async_operation_source* async)
  {
    //Validate inputs
    options.Validate();

    //Load controlnet
    if (async) async->update_state("Loading controlnet...");
    EnsureControlNet(options.ConditionType);

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

    //Schedule steps
    list<Tensor> derivatives;
    auto steps = context.Scheduler.GetSteps(options.StepCount);
    auto initialStep = size_t(clamp(int(options.StepCount - options.StepCount * options.DenoisingStrength - 1), 0, int(options.StepCount)));

    //Create initial sample
    auto latentSample = options.LatentInput ? StableDiffusionInferer::PrepareLatentSample(context, options.LatentInput, steps.Sigmas[initialStep]) : StableDiffusionInferer::GenerateLatentSample(context);

    //Bind constant inputs / outputs  
    IoBinding controlnetBinding{ _controlnetSession };
    /*for (auto i = 0; i < 12; i++)
    {
      controlnetBinding.BindOutput(format("down_block_{}_additional_residual", i).c_str(), _environment.MemoryInfo());
    }
    controlnetBinding.BindOutput("mid_block_additional_residual", _environment.MemoryInfo());*/
    {
      controlnetBinding.BindOutput("down_block_res_samples", _environment.MemoryInfo());
      controlnetBinding.BindOutput("mid_block_res_sample", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19820", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19822", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19824", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19826", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19828", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19830", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19832", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19834", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19836", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19838", _environment.MemoryInfo());
      controlnetBinding.BindOutput("19840", _environment.MemoryInfo());
    }

    controlnetBinding.BindInput("controlnet_cond", options.ConditionInput/*.ToHalf()*/.ToOrtValue());
    controlnetBinding.BindInput("conditioning_scale", Tensor(double(options.ConditioningScale)).ToOrtValue());
    
    IoBinding unetBinding{ _unetSession };
    unetBinding.BindOutput("out_sample", _environment.MemoryInfo());

    if (holds_alternative<Tensor>(options.TextEmbeddings))
    {
      auto encoderHiddenStates = get<Tensor>(options.TextEmbeddings)/*.ToHalf()*/.Duplicate(options.BatchSize);
      controlnetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());
      unetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());
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
          auto encoderHiddenStates = currentEmbedding->/*ToHalf().*/Duplicate(options.BatchSize);
          controlnetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());
          unetBinding.BindInput("encoder_hidden_states", encoderHiddenStates.ToOrtValue());
        }
      }

      //Update sample
      {
        auto scaledSample = (latentSample.Duplicate().Swizzle(options.BatchSize) / sqrt(steps.Sigmas[i] * steps.Sigmas[i] + 1))/*.ToHalf()*/;
        controlnetBinding.BindInput("sample", scaledSample.ToOrtValue());
        unetBinding.BindInput("sample", scaledSample.ToOrtValue());
      }

      //Update timestep
      {
        auto timestep = Tensor(steps.Timesteps[i])/*.ToHalf()*/;
        controlnetBinding.BindInput("timestep", timestep.ToOrtValue());
        unetBinding.BindInput("timestep", timestep.ToOrtValue());
      }

      //Run ControlNet
      _controlnetSession.Run({}, controlnetBinding);

      //Read outputs and feed them into UNet
      {
        auto controlnetOutputs = controlnetBinding.GetOutputValues();
        for (auto i = 0; i < 12; i++)
        {
          //unetBinding.BindInput(format("down_block_{}_additional_residual", i).c_str(), controlnetOutputs[i]);
          auto x = Tensor::FromOrtValue(controlnetOutputs[i]);
          auto y = x.AsSpan<float>();

          unetBinding.BindInput(format("down_block_{}", i).c_str(), controlnetOutputs[i]);
          printf("asd");
        }
        unetBinding.BindInput("mid_block_additional_residual", controlnetOutputs.back());
      }

      //Run UNet
      _unetSession.Run({}, unetBinding);

      //Read output
      auto outputs = unetBinding.GetOutputValues();
      auto output = Tensor::FromOrtValue(outputs[0])/*.ToSingle()*/;


      auto outputComponents = output.Swizzle().Split();

      //Calculate guidance
      auto& blankNoise = outputComponents[0];
      auto& textNoise = outputComponents[1];
      auto guidedNoise = blankNoise.BinaryOperation<float>(textNoise, [guidanceScale = options.GuidanceScale](float a, float b)
        { return a + guidanceScale * (b - a); });

      //Refine latent image
      latentSample = steps.ApplyStep(latentSample, guidedNoise, derivatives, context.Randoms, i);

      auto y = latentSample.AsSpan<float>();

      //Apply mask
      if (options.MaskInput)
      {
        auto maskedSample = StableDiffusionInferer::PrepareLatentSample(context, options.LatentInput, steps.Sigmas[i]);
        latentSample = StableDiffusionInferer::BlendLatentSamples(maskedSample, latentSample, options.MaskInput);
      }
    }

    //Decode sample
    latentSample = latentSample * (1.0f / 0.18215f);
    return latentSample;
  }

  void ControlNetInferer::EnsureControlNet(ControlNetType type)
  {
    if (_controlnetType == type) return;

    _controlnetSession = Session{ nullptr };

    auto modelPath = _controlnetPath / format(L"{}.onnx", ToString(type));
    _controlnetSession = _environment.CreateSession(modelPath);
    _controlnetType = type;
  }

  const wchar_t* ControlNetInferer::ToString(ControlNetType type)
  {
    switch (type)
    {
    case ControlNetType::Canny:
      return L"canny";
    case ControlNetType::Hed:
      return L"hed";
    case ControlNetType::Depth:
      return L"depth";
    default:
      throw logic_error("ControlNet type not implemented.");
    }
  }

  void ControlNetOptions::Validate() const
  {
    StableDiffusionOptions::Validate();
    //TBD
  }
}