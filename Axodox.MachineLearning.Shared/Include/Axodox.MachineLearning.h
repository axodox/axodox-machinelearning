#pragma once
#include "../includes.h"

//General
#include "MachineLearning/OnnxEnvironment.h"
#include "MachineLearning/OnnxExtensions.h"
#include "MachineLearning/OnnxModelMetadata.h"
#include "MachineLearning/OnnxModelStatistics.h"

//Image generation
#include "MachineLearning/Prompts/PromptAttention.h"
#include "MachineLearning/Prompts/PromptParser.h"
#include "MachineLearning/Prompts/PromptScheduler.h"

#include "MachineLearning/Schedulers/StableDiffusionScheduler.h"
#include "MachineLearning/Schedulers/EulerAncestralScheduler.h"
#include "MachineLearning/Schedulers/DpmPlusPlus2MScheduler.h"

#include "MachineLearning/SafetyChecker.h"
#include "MachineLearning/SafetyCheckerOptions.h"
#include "MachineLearning/StableDiffustionInferer.h"
#include "MachineLearning/TensorInfo.h"
#include "MachineLearning/Tensor.h"
#include "MachineLearning/TensorType.h"
#include "MachineLearning/TextEmbedder.h"
#include "MachineLearning/TextEncoder.h"
#include "MachineLearning/TextTokenizer.h"
#include "MachineLearning/VaeDecoder.h"
#include "MachineLearning/VaeEncoder.h"

#include "MachineLearning/ControlNetInferer.h"

//Depth estimation
#include "MachineLearning/DepthEstimator.h"
#include "MachineLearning/EdgeDetector.h"
#include "MachineLearning/PoseEstimator.h"

//Model downloads
#ifdef PLATFORM_WINDOWS
#include "Web/HuggingFaceClient.h"
#include "Web/HuggingFaceSchema.h"
#endif