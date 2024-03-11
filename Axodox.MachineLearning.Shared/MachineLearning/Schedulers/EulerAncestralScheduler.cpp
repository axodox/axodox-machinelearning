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
        auto currentSigma = _sigmas[step];
        auto nextSigma = _sigmas[step + 1];

        Tensor predictedOriginalSample;

        if (_predictiontype == StableDiffusionSchedulerPredictionType::V) 
        {

            predictedOriginalSample = output.BinaryOperation<float>(input, [currentSigma](float model_output, float sample) {
                float sigmaSquaredPlusOne = currentSigma * currentSigma + 1;
                return (model_output * (-currentSigma / std::sqrt(sigmaSquaredPlusOne))) + (sample / sigmaSquaredPlusOne); // note: std::sqrt is VITAL here (???)
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
         

        // Calculate sigma squared values for the process
        auto currentSigmaSquared = currentSigma * currentSigma;
        auto nextSigmaSquared = nextSigma * nextSigma;

        // Calculate sigma_up and sigma_down according to the Python logic
        float sigmaUp = std::sqrt(max(0.0f, nextSigmaSquared - currentSigmaSquared));
        float sigmaDown = std::sqrt(nextSigmaSquared - sigmaUp * sigmaUp);

        // Calculate dt based on sigma changes
        float dt = sigmaDown - currentSigma;

        // Derivative calculation (the 'derivative' here is conceptual, representing the reverse diffusion step)
        auto derivative = input.BinaryOperation<float>(predictedOriginalSample, [currentSigma](float inputVal, float predOriginalVal) {
            return (inputVal - predOriginalVal) / currentSigma;
            });

        // Update sample with derivative and dt
        auto updatedSample = input.BinaryOperation<float>(derivative, [dt](float inputVal, float derivativeVal) {
            return inputVal + derivativeVal * dt;
            });

        // Generate random noise scaled by sigmaUp
        auto randomNoise = Tensor::CreateRandom(input.Shape, _randoms, sigmaUp);

        // Add noise to the updated sample
        updatedSample = updatedSample.BinaryOperation<float>(randomNoise, [](float updatedSampleVal, float noiseVal) {
            return updatedSampleVal + noiseVal;
            });

        return updatedSample;
    }
}