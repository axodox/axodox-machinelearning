#pragma once

#ifdef PLATFORM_WINDOWS
#include <winrt/windows.foundation.h>
#include <winrt/windows.web.http.h>
#include <winrt/windows.web.http.filters.h>
#include <winrt/windows.web.http.headers.h>
#include <winrt/windows.storage.h>
#include <winrt/windows.storage.streams.h>
#endif

#ifdef AXODOX_MACHINELEARNING_EXPORT
#define USE_ONNX
#endif

#ifdef USE_ONNX
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"
#endif

#ifdef AXODOX_MACHINELEARNING_EXPORT
#define AXODOX_MACHINELEARNING_API __declspec(dllexport)
#else
#define AXODOX_MACHINELEARNING_API __declspec(dllimport)

#ifdef PLATFORM_WINDOWS
#pragma comment (lib,"Axodox.MachineLearning.lib")
#endif
#endif