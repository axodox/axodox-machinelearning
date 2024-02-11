#pragma once
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  enum class StableDiffusionBetaSchedulerKind
  {
    ScaledLinear,
    Linear
  };

  enum class StableDiffusionPredictorKind
  {
    Epsilon,
    //VPrediction
  };

  enum class StableDiffusionSchedulerKind
  {
    LmsDiscrete,
    EulerAncestral,
    DpmPlusPlus2M
  };

  struct AXODOX_MACHINELEARNING_API StableDiffusionSchedulerOptions
  {
    size_t TrainStepCount = 1000;
    float BetaAtStart = 0.00085f;
    float BetaAtEnd = 0.012f;
    StableDiffusionBetaSchedulerKind BetaSchedulerType = StableDiffusionBetaSchedulerKind::ScaledLinear;
    StableDiffusionPredictorKind PredictorType = StableDiffusionPredictorKind::Epsilon;
    StableDiffusionSchedulerKind SchedulerType = StableDiffusionSchedulerKind::DpmPlusPlus2M;
    std::vector<float> BetasTrained;
  };

  typedef std::vector<float> LmsCoefficients;

  struct AXODOX_MACHINELEARNING_API EulerCoefficients
  {
    float SigmaDown, SigmaUp;
  };

  typedef std::variant<LmsCoefficients, EulerCoefficients> coefficient_t;
  typedef std::vector<coefficient_t> coefficients_t;

  struct AXODOX_MACHINELEARNING_API StableDiffusionSchedulerSteps
  {
    static const size_t DerivativeOrder;

    std::vector<float> Timesteps;
    std::vector<float> Sigmas;
    coefficients_t Coefficients;
    StableDiffusionSchedulerKind SchedulerType;

    Tensor ApplyStep(const Tensor& latents, const Tensor& noise, std::list<Tensor>& derivatives, std::span<std::minstd_rand> randoms, size_t step);
  };

  class AXODOX_MACHINELEARNING_API StableDiffusionScheduler
  {
  public:
    StableDiffusionScheduler(const StableDiffusionSchedulerOptions& options = {});

    StableDiffusionSchedulerSteps GetSteps(size_t count) const;

    float InitialNoiseSigma() const;
    std::span<const float> CumulativeAlphas() const;

  private:
    StableDiffusionSchedulerOptions _options;
    std::vector<float> _cumulativeAlphas;
    float _initialNoiseSigma;

    std::vector<float> GetLinearBetas() const;
    std::vector<float> GetScaledLinearBetas() const;
    static std::vector<float> CalculateCumulativeAlphas(std::span<const float> betas);
    static float CalculateInitialNoiseSigma(std::span<const float> cumulativeAlphas);
    
    static LmsCoefficients GetLmsCoefficients(size_t step, std::span<const float> sigmas);
    static EulerCoefficients GetEulerCoefficients(size_t step, std::span<const float> sigmas);
    static void ApplyKarrasSigmas(std::span<float> sigmas);
  };
}