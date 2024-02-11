#include "pch.h"
#include "StableDiffusionScheduler2.h"
#include "EulerAncestralScheduler.h"
#include "DpmPlusPlus2MScheduler.h"

using namespace std;

namespace Axodox::MachineLearning
{
  StableDiffusionScheduler2::StableDiffusionScheduler2(const StableDiffusionSchedulerOptions2& options) :
    _randoms(options.Randoms)
  {
    //Betas - scaled linear
    vector<float> betas;
    if (options.BetasTrained.empty())
    {
      betas.resize(options.TrainStepCount);

      auto value = sqrt(options.BetaAtStart);
      auto step = (sqrt(options.BetaAtEnd) - value) / (options.TrainStepCount - 1.f);
      for (auto& beta : betas)
      {
        beta = value * value;
        value += step;
      }
    }
    else
    {
      if (betas.size() != options.TrainStepCount) throw invalid_argument("options.BetasTrained.Size() != options.TrainStepCount");

      betas = { options.BetasTrained.begin(), options.BetasTrained.end() };
    }

    //Cumulative alphas
    vector<float> cumulativeAlphas{ betas.begin(), betas.end() };
    {
      float value = 1.f;
      for (auto& result : cumulativeAlphas)
      {
        value *= 1.f - result;
        result = value;
      }
    }

    //Timesteps
    vector<float> timesteps;
    {
      timesteps.resize(options.InferenceStepCount);

      auto stepRatio = options.TrainStepCount / options.InferenceStepCount;
      auto step = (options.TrainStepCount - 1) / float(options.InferenceStepCount - 1);
      for (auto value = 0; auto & timestep : timesteps)
      {
        timestep = value++ * stepRatio + 1;
      }

      ranges::reverse(timesteps);
    }

    //Training sigmas
    vector<float> trainingSigmas{ cumulativeAlphas.begin(), cumulativeAlphas.end() };
    for (auto& sigma : trainingSigmas)
    {
      sigma = sqrt((1.f - sigma) / sigma);
    }

    //Inference sigmas
    vector<float> inferenceSigmas;
    {
      inferenceSigmas.reserve(options.InferenceStepCount + 1);
      for (auto trainStep : timesteps)
      {
        auto previousIndex = max(size_t(floor(trainStep)), size_t(0));
        auto nextIndex = min(size_t(ceil(trainStep)), trainingSigmas.size() - 1);
        inferenceSigmas.push_back(lerp(trainingSigmas[previousIndex], trainingSigmas[nextIndex], trainStep - floor(trainStep)));
      }
      inferenceSigmas.push_back(0.f);
    }

    //Store results
    _trainingSigmas = move(trainingSigmas);
    _sigmas = move(inferenceSigmas);
    _timesteps = move(timesteps);
  }

  std::unique_ptr<StableDiffusionScheduler2> StableDiffusionScheduler2::Create(StableDiffusionSchedulerKind2 kind, const StableDiffusionSchedulerOptions2& options)
  {
    switch (kind)
    {
    case StableDiffusionSchedulerKind2::EulerAncestral:
      return make_unique<EulerAncestralScheduler>(options);
    case StableDiffusionSchedulerKind2::DpmPlusPlus2M:
      return make_unique<DpmPlusPlus2MScheduler>(options);
    default:
      return nullptr;
    }
  }

  std::span<const float> StableDiffusionScheduler2::Timesteps() const
  {
    return _timesteps;
  }

  std::span<const float> StableDiffusionScheduler2::Sigmas() const
  {
    return _sigmas;
  }

  float StableDiffusionScheduler2::SigmaToTime(float sigma) const
  {
    auto TIMESTEPS = int(_timesteps.size());

    int low_idx = TIMESTEPS - 1, high_idx = TIMESTEPS;
    for (auto i = 0; i < _trainingSigmas.size(); i++) {
      if (_trainingSigmas[i] >= sigma)
      {
        low_idx = max(0, i - 1);
        high_idx = low_idx + 1;
        break;
      }
    }

    float low = log(_trainingSigmas[low_idx]);
    float high = log(_trainingSigmas[high_idx]);
    float w = (low -log(sigma)) / (low - high);
    w = max(0.f, min(1.f, w));
    float t = (1.0f - w) * low_idx + w * high_idx;

    return t;
  }
}