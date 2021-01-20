import transformers
from transformers import PretrainedConfig, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, TOKENIZER_MAPPING, CONFIG_MAPPING
from fastseq.models.unilm_hf.configuration_unilm import UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP, UnilmConfig
from fastseq.models.unilm_hf.modeling_unilm import UnilmForSeq2Seq
from fastseq.models.unilm_hf.tokenization_unilm import UnilmTokenizer
from fastseq.utils.api_decorator import replace

CONFIG_MAPPING['unilm'] = UnilmConfig
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[UnilmConfig] = UnilmForSeq2Seq
TOKENIZER_MAPPING[UnilmConfig] = (UnilmTokenizer, None)
TOKENIZER_MAPPING.move_to_end(transformers.configuration_bert.BertConfig)


@replace(AutoModelForSeq2SeqLM)
class AutoModelForSeq2SeqLMV2(AutoModelForSeq2SeqLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            if pretrained_model_name_or_path in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP:
                pretrained_config_name_or_path = UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP[
                    pretrained_model_name_or_path]
            else:
                pretrained_config_name_or_path = pretrained_model_name_or_path
            config = AutoConfig.from_pretrained(pretrained_config_name_or_path,
                                                **kwargs)

        for config_class, model_class in transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.items(
        ):
            if isinstance(config, config_class):
                return model_class.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in transformers.
                          MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()),
            ))


@replace(AutoTokenizer)
class AutoTokenizerV2(AutoTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            if pretrained_model_name_or_path in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP:
                pretrained_config_name_or_path = UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP[
                    pretrained_model_name_or_path]
            else:
                pretrained_config_name_or_path = pretrained_model_name_or_path
            config = AutoConfig.from_pretrained(pretrained_config_name_or_path,
                                                **kwargs)

        if "bert-base-japanese" in str(pretrained_model_name_or_path):
            return BertJapaneseTokenizer.from_pretrained(
                pretrained_model_name_or_path, *inputs, **kwargs)

        use_fast = kwargs.pop("use_fast", False)
        for config_class, (tokenizer_class_py, tokenizer_class_fast
                           ) in transformers.TOKENIZER_MAPPING.items():
            if isinstance(config, config_class):
                if tokenizer_class_fast and use_fast:
                    return tokenizer_class_fast.from_pretrained(
                        pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    return tokenizer_class_py.from_pretrained(
                        pretrained_model_name_or_path, *inputs, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} to build an AutoTokenizer.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                ", ".join(c.__name__ for c in TOKENIZER_MAPPING.keys())))
