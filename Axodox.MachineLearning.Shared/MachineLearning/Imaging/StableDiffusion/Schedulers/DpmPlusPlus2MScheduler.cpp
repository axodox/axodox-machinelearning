#include "pch.h"
#include "DpmPlusPlus2MScheduler.h"

using namespace std;

namespace Axodox::MachineLearning::Imaging::StableDiffusion::Schedulers
{
  DpmPlusPlus2MScheduler::DpmPlusPlus2MScheduler(const StableDiffusionSchedulerOptions& options) :
    StableDiffusionScheduler(options)
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

    _sigmas = move(sigmas);
    _timesteps = move(timesteps);

    //_sigmas = { 14.614643096923828f, 10.81978988647461f, 7.9029436111450195f, 5.687837600708008f, 4.027743816375732f, 2.8015711307525635f, 1.9103621244430542f, 1.2741049528121948f,  0.8288719058036804f, 0.524259626865387f, 0.3211216926574707f, 0.18956425786018372f, 0.10719937086105347f, 0.05763240531086922f, 0.02916753850877285f, 0.f };
    //_timesteps = { 999.f, 947.6224f, 889.5464f, 823.0464f, 745.8676f, 655.3113f, 549.0170f, 427.4898f, 298.6582f, 179.8307f, 89.9427f, 36.5918f, 12.0011f, 2.8839f, 0.f };
  }

  Tensor DpmPlusPlus2MScheduler::ApplyStep(const Tensor& input, const Tensor& output, size_t step)
  {
    auto currentSigma = _sigmas[step];
    auto nextSigma = _sigmas[step + 1];

    auto predictedOriginalSample = input.BinaryOperation<float>(output, [currentSigma](float a, float b) { return a - currentSigma * b; });

    float t = -log(currentSigma);
    float tNext = -log(nextSigma);
    float h = tNext - t;

    Tensor denoised;
    if (!_previousPredictedSample || nextSigma == 0)
    {
      denoised = predictedOriginalSample;
    }
    else
    {
      float hLast = t - -log(_sigmas[step - 1]);
      float r = hLast / h;

      auto x = 1.f + 1.f / (2.f * r);
      auto y = 1.f / (2.f * r);

      denoised = predictedOriginalSample.BinaryOperation<float>(_previousPredictedSample, [=](float a, float b) {
        return x * a - y * b;
        });
    }

    if (nextSigma != 0)
    {
      _previousPredictedSample = predictedOriginalSample;
    }
    else
    {
      _previousPredictedSample.Reset();
    }

    float x = nextSigma / currentSigma;
    float y = exp(-h) - 1.f;
    return input.BinaryOperation<float>(denoised, [=](float a, float b) {
      return a * x - b * y;
      });
  }
}