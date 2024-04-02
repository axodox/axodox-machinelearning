#pragma once

#ifdef PLATFORM_WINDOWS
#include <Windows.h>
#include <winrt/windows.foundation.h>
#include <winrt/windows.web.http.h>
#include <winrt/windows.web.http.filters.h>
#include <winrt/windows.web.http.headers.h>
#include <winrt/windows.storage.h>
#include <winrt/windows.storage.streams.h>
#include <directml.h>
#include "dml_provider_factory.h"
#endif

#include "onnxruntime_cxx_api.h"
#include "Include/Axodox.Infrastructure.h"
#include "Include/Axodox.Storage.h"
#include "Include/Axodox.Graphics.h"
#include "Include/Axodox.Threading.h"

#ifdef AXODOX_MACHINELEARNING_EXPORT
#define AXODOX_MACHINELEARNING_API __declspec(dllexport)
#else
#define AXODOX_MACHINELEARNING_API __declspec(dllimport)

#ifdef PLATFORM_WINDOWS
#pragma comment (lib,"Axodox.MachineLearning.lib")
#endif

#endif