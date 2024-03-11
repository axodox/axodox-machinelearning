#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
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
    TextEncoder(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor EncodeText(const Tensor& text);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;

    bool _has64bitInputIds;
    bool _hasHiddenLayers;

  protected:
      friend class TextEncodingProvider;
      bool isSDXL;

  };

  class AXODOX_MACHINELEARNING_API TextEncoder2
  {
    static inline const Infrastructure::logger _logger{ "TextEncoder2" };

  public:
    TextEncoder2(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    EncodedText EncodeText(const Tensor& text);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;

    bool _has64bitInputIds;
  };

  class AXODOX_MACHINELEARNING_API TextEncodingProvider
  {
  public:
    TextEncodingProvider(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    EncodedText EncodeText(const Tensor& text);

  private:
    TextEncoder _textEncoder;
    std::unique_ptr<TextEncoder2> _textEncoder2;
  };
}