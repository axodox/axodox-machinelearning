#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  struct AXODOX_MACHINELEARNING_API EncodedText
  {
    Tensor LastHiddenState;
    Tensor TextEmbeds;

    EncodedText Concat(const EncodedText& other) const;
  };

  class AXODOX_MACHINELEARNING_API TextEncoder
  {
    static inline const Infrastructure::logger _logger{ "TextEncoder" };

  public:
    TextEncoder(const Sessions::OnnxSessionParameters& parameters);

    Tensor EncodeText(const Tensor& text);

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
    bool _has64bitInputIds;
    bool _hasHiddenLayers;
  };

  class AXODOX_MACHINELEARNING_API TextEncoder2
  {
    static inline const Infrastructure::logger _logger{ "TextEncoder2" };

  public:
    TextEncoder2(const Sessions::OnnxSessionParameters& parameters);

    EncodedText EncodeText(const Tensor& text);

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
    bool _has64bitInputIds;
  };

  class AXODOX_MACHINELEARNING_API TextEncodingProvider
  {
  public:
    TextEncodingProvider(
      const Sessions::OnnxSessionParameters& encoder1Parameters, 
      const Sessions::OnnxSessionParameters& encoder2Parameters);

    EncodedText EncodeText(const Tensor& text);

  private:
    TextEncoder _textEncoder;
    std::unique_ptr<TextEncoder2> _textEncoder2;
  };
}