## Introduction

This repository contains a **fully C++ implementation of Stable Diffusion**-based image synthesis, including the original txt2img, img2img and inpainting capabilities and the safety checker. This solution **does not depend on Python** and **runs the entire image generation process in a single process with competitive performance**, making deployments significantly simpler and smaller, essentially consisting a few executable and library files, and the model weights. Using the library it is possible to integrate Stable Diffusion into almost any application - as long as it can import C++ or C functions, but it is **most useful for the developers of realtime graphics applications and games**, which are often realized with C++.

<table style="margin: 0px auto;">
  <tr>
    <td><img src="https://media.githubusercontent.com/media/axodox/unpaint/main/Unpaint/Showcase/2023-06-03%2020-50-21.png" alt="a samurai drawing his sword to defend his land" width="256" height="256"></td>
    <td><img src="https://media.githubusercontent.com/media/axodox/unpaint/main/Unpaint/Showcase/2023-06-03%2020-48-40.png" alt="a sailship crossing the high sea, 18st century, impressionist painting, closeup" width="256" height="256"></td>
    <td><img src="https://media.githubusercontent.com/media/axodox/unpaint/main/Unpaint/Showcase/2023-06-03%2019-32-26.png" alt=" close up portrait photo of woman in wastelander clothes, long haircut, pale skin, slim body, background is city ruins, (high detailed skin:1.2)" width="256" height="256"></td>
  </tr>
</table>

## Technical background

The implementation uses the [ONNX](https://onnx.ai/) to store the mathematical models involved in the image generation. These ONNX models are then executed using the [ONNX runtime](https://github.com/microsoft/onnxruntime), which support a variety of platforms (Windows, Linux, MacOS, Android, iOS, WebAssembly etc.), and execution providers (such as NVIDIA CUDA / TensorRT; AMD ROCm, Apple CoreML, Qualcomm QNN, Microsoft DirectML and many more). 

We provide an example integration called [Unpaint](https://github.com/axodox/unpaint) which showcases how the libraries can be integrated in a simple WinUI based user interface. You may download a recent release from [here](https://github.com/axodox/unpaint/releases) to evaluate the performance characteristics of the solution. 

> Please note that you will need to install the provided test certificate to run the example app into your Trusted Root Certificate Authorities Store (in the local machine container), you can also [resign the package with your own certificate](https://learn.microsoft.com/en-us/windows/msix/package/create-certificate-package-signing) to make it install.

The current codebase and the resulting [Nuget packages](https://www.nuget.org/packages/Axodox.MachineLearning) target Windows and use DirectML, however only small sections of the code utilize Windows specific APIs, and thus could be ported to other platforms with minimal effort.

## Licensing

The source code of this library is provided under the MIT license.

## Integrating the component

Prebuilt versions of the project can be retrieved from Nuget under the name `Axodox.MachineLearning` and added to Visual Studio C++ projects (both desktop and UWP projects are supported) with the x64 platform.

Basic integration:
- Add the `Axodox.Common` and `Axodox.MachineLearning` packages to your project
- Ensure that your compiler is set to **C++20**, we also recommend enabling all warnings and conformance mode
- Add the following include statement to your code file or precompiled header: `#include "Include/Axodox.MachineLearning.h"`
- Follow this example code to integrate the pipeline: https://github.com/axodox/unpaint/blob/main/Unpaint/StableDiffusionModelExecutor.cpp

> We recommend adding appropriate safety mechanisms to your app to suppress inappropriate outputs of StableDiffusion, the performance overhead is insignificant.

# Building the project

Building the library is required to make and test changes. You will need to have the following installed to build the library:

- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
  - Select the following workloads:
    - Desktop development with C++    
    - Game development with C++
  - To build [Unpaint](https://github.com/axodox/unpaint) as well also select these individual packages:
    - Universal Windows Platform development
    - C++ (v143) Universal Windows Platform tools

You can either run `build_nuget.ps1` or open `Axodox.MachineLearning.sln` and build from Visual Studio.

Once you have built the library, you override your existing nuget package install by setting the `AxodoxMachineLearning-Location` environment variable to point to your local build. 

> For example `C:\dev\axodox-machinelearning\Axodox.MachineLearning.Universal` for an UWP app and `C:\dev\axodox-machinelearning\Axodox.MachineLearning.Desktop` for a desktop app.

This allows to add all projects into the same solution and make changes on the library and your app seamlessly without copying files repeatedly.
