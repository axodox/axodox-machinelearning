#include "pch.h"
#include "EulerAncestralScheduler.h"

using namespace std;

namespace Axodox::MachineLearning
{
  EulerAncestralScheduler::EulerAncestralScheduler(const StableDiffusionSchedulerOptions& options) : 
    StableDiffusionScheduler(options)
  { }

  Tensor EulerAncestralScheduler::ApplyStep(const Tensor& input, const Tensor& output, size_t step)
  {
    //x = input
    //denoised = predictedOriginalSample
    //d => currentDerivative aka output
    auto currentSigma = _sigmas[step];
    auto nextSigma = _sigmas[step + 1];

    //Compute predicted original sample (x_0) from sigma-scaled predicted noise
    auto predictedOriginalSample = input.BinaryOperation<float>(output, [currentSigma](float a, float b) { return a - currentSigma * b; });

    //currentDerivative == output == d
    //auto currentDerivative = input.BinaryOperation<float>(predictedOriginalSample, [sigma](float a, float b) { return (a - b) / sigma; });

    //Get ancestral step
    auto currentSigmaSquared = currentSigma * currentSigma;
    auto nextSigmaSquared = nextSigma * nextSigma;
    
    float sigmaUp = min(nextSigma,
      sqrt((currentSigmaSquared - nextSigmaSquared) * nextSigmaSquared / currentSigmaSquared));
    float sigmaDown = sqrt(nextSigmaSquared - sigmaUp * sigmaUp);

    //Euler method
    float dt = sigmaDown - currentSigma;
    auto randomNoise = Tensor::CreateRandom(input.Shape, _randoms, sigmaUp);
    auto latentDelta = randomNoise.BinaryOperation<float>(output, [dt](float a, float b) { return a + dt * b; });
    return input.BinaryOperation<float>(latentDelta, [](float a, float b) { return a + b; });
  }
}