#pragma once
#include "../includes.h"

//General
#include "MachineLearning/Tensor.h"
#include "MachineLearning/TensorInfo.h"
#include "MachineLearning/TensorType.h"
#include "MachineLearning/OnnxExtensions.h"

//Sessions
#include "MachineLearning/Sessions/OnnxEnvironment.h"
#include "MachineLearning/Sessions/OnnxModelSource.h"
#include "MachineLearning/Sessions/OnnxModelMetadata.h"
#include "MachineLearning/Sessions/OnnxModelStatistics.h"
#include "MachineLearning/Sessions/OnnxSession.h"

//Annotators
#include "MachineLearning/Imaging/Annotators/ImageAnnotator.h"
#include "MachineLearning/Imaging/Annotators/DepthEstimator.h"
#include "MachineLearning/Imaging/Annotators/EdgeDetector.h"
#include "MachineLearning/Imaging/Annotators/PoseEstimator.h"

//Stable Diffusion
#include "MachineLearning/Imaging/StableDiffusion/TextTokenizer.h"
#include "MachineLearning/Imaging/StableDiffusion/TextEncoder.h"
#include "MachineLearning/Imaging/StableDiffusion/TextEmbedder.h"
#include "MachineLearning/Imaging/StableDiffusion/VaeEncoder.h"
#include "MachineLearning/Imaging/StableDiffusion/StableDiffustionInferer.h"
#include "MachineLearning/Imaging/StableDiffusion/ControlNetInferer.h"
#include "MachineLearning/Imaging/StableDiffusion/VaeDecoder.h"
#include "MachineLearning/Imaging/StableDiffusion/SafetyCheckerOptions.h"
#include "MachineLearning/Imaging/StableDiffusion/SafetyChecker.h"

//Stable Diffusion - schedulers
#include "MachineLearning/Imaging/StableDiffusion/Schedulers/StableDiffusionScheduler.h"
#include "MachineLearning/Imaging/StableDiffusion/Schedulers/EulerAncestralScheduler.h"
#include "MachineLearning/Imaging/StableDiffusion/Schedulers/DpmPlusPlus2MScheduler.h"

//Munkres
#include "MachineLearning/Solvers/Munkres/CostGraph.h"
#include "MachineLearning/Solvers/Munkres/CoverTable.h"
#include "MachineLearning/Solvers/Munkres/MunkresSolver.h"
#include "MachineLearning/Solvers/Munkres/PairGraph.h"

//Prompts
#include "MachineLearning/Text/Prompts/PromptAttention.h"
#include "MachineLearning/Text/Prompts/PromptParser.h"
#include "MachineLearning/Text/Prompts/PromptScheduler.h"
#include "MachineLearning/Text/Prompts/PromptSplitter.h"

//Model downloads
#ifdef PLATFORM_WINDOWS
#include "Web/HuggingFaceClient.h"
#include "Web/HuggingFaceSchema.h"
#endif