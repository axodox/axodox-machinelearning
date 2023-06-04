#pragma once
#include "../includes.h"

#include "MachineLearning/Prompts/PromptAttention.h"
#include "MachineLearning/Prompts/PromptParser.h"
#include "MachineLearning/Prompts/PromptScheduler.h"

#include "MachineLearning/OnnxEnvironment.h"
#include "MachineLearning/OnnxExtensions.h"
#include "MachineLearning/OnnxModelStatistics.h"
#include "MachineLearning/SafetyChecker.h"
#include "MachineLearning/SafetyCheckerOptions.h"
#include "MachineLearning/StableDiffusionScheduler.h"
#include "MachineLearning/StableDiffustionInferer.h"
#include "MachineLearning/Tensor.h"
#include "MachineLearning/TensorType.h"
#include "MachineLearning/TextEmbedder.h"
#include "MachineLearning/TextEncoder.h"
#include "MachineLearning/TextTokenizer.h"
#include "MachineLearning/VaeDecoder.h"
#include "MachineLearning/VaeEncoder.h"

#ifdef PLATFORM_WINDOWS
#include "Web/HuggingFaceClient.h"
#include "Web/HuggingFaceSchema.h"
#endif