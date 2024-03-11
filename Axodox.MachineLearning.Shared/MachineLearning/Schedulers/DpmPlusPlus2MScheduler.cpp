#include "pch.h"
#include "DpmPlusPlus2MScheduler.h"

using namespace std;

namespace Axodox::MachineLearning
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

    }

    Tensor DpmPlusPlus2MScheduler::ApplyStep(const Tensor& input, const Tensor& output, size_t step)
    {
        auto currentSigma = _sigmas[step];
        auto nextSigma = _sigmas[step + 1];

        Tensor predictedOriginalSample;

        if (_predictiontype == StableDiffusionSchedulerPredictionType::V) 
        {

            predictedOriginalSample = output.BinaryOperation<float>(input, [currentSigma](float model_output, float sample) {
                float sigmaSquaredPlusOne = currentSigma * currentSigma + 1;
                return (model_output * (-currentSigma / std::sqrt(sigmaSquaredPlusOne))) + (sample / sigmaSquaredPlusOne);
                });
        
        }
        else if (_predictiontype == StableDiffusionSchedulerPredictionType::Epsilon)
        {
            predictedOriginalSample = input.BinaryOperation<float>(output, [currentSigma](float a, float b) { return a - currentSigma * b; });

        }
        else
        {
            throw std::invalid_argument("Uninmplemented prediction type.");

        }

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
