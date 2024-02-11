#include "pch.h"
#include "DpmPlusPlus2MScheduler.h"

using namespace std;

namespace Axodox::MachineLearning
{
  DpmPlusPlus2MScheduler::DpmPlusPlus2MScheduler(const StableDiffusionSchedulerOptions2& options) :
    StableDiffusionScheduler2(options)
  {
    //Apply Karras sigmas
    const auto rho = 7.f;

    auto sigmaMax = _sigmas.front();
    auto sigmaMin = *(_sigmas.end() - 2);

    auto invRhoMin = pow(sigmaMin, 1.f / rho);
    auto invRhoMax = pow(sigmaMax, 1.f / rho);

    auto stepCount = _sigmas.size() - 1;
    auto stepSize = 1.f / (stepCount - 1);
    vector<float> timesteps(_timesteps.size());
    vector<float> sigmas(_sigmas.size());
    for (auto i = 0; i < stepCount; i++)
    {
      auto t = i * stepSize;
      sigmas[i] = pow(invRhoMax + t * (invRhoMin - invRhoMax), rho);
      timesteps[i] = SigmaToTime(sigmas[i]);
    }

    vector<float> timestepsInterpol(_timesteps.size());
    vector<float> sigmasInterpol(_sigmas.size());
    for (auto i = 0; i < stepCount; i++)
    {
      sigmasInterpol[i] = (sigmas[i] + sigmas[i + 1]) / 2.f;
      timestepsInterpol[i] = SigmaToTime(sigmasInterpol[i]);
    }

    //_timesteps = { 999.f, 947.6224f, 889.5464f, 823.0464f, 745.8676f, 655.3113f, 549.0170f, 427.4898f, 298.6582f, 179.8307f, 89.9427f, 36.5918f, 12.0011f, 2.8839f, 0.f };
  }

  Tensor DpmPlusPlus2MScheduler::ApplyStep(const Tensor& input, const Tensor& output, size_t step)
  {
    //x = input
    //denoised = predictedOriginalSample
    //old_denoised = _previousPredictedSample
    //d => currentDerivative aka output

    auto currentSigma = _sigmas[step];
    auto nextSigma = _sigmas[step + 1];

    auto predictedOriginalSample = input.BinaryOperation<float>(output, [currentSigma](float a, float b) { return a - currentSigma * b; });

    float t = -log(currentSigma);
    float t_next = -log(nextSigma);
    float h = t_next - t;

    Tensor denoised;
    if (step == 0 || nextSigma == 0)
    {
      denoised = predictedOriginalSample;
    }
    else
    {
      float h_last = t - -log(_sigmas[step - 1]);
      float r = h_last / h;

      auto x = 1.f + 1.f / (2.f * r);
      auto y = 1.f / (2.f * r);

      denoised = predictedOriginalSample.BinaryOperation<float>(_previousPredictedSample, [=](float a, float b) {
        return x * a - y * b;
        });
    }

    _previousPredictedSample = predictedOriginalSample;


    float x = nextSigma / currentSigma;
    float y = exp(-h) - 1.f;
    return input.BinaryOperation<float>(denoised, [=](float a, float b) {
      return a * x - b * y;
      });
  }
}