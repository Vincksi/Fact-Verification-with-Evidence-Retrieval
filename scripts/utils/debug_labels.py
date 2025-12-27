from transformers import AutoConfig
config = AutoConfig.from_pretrained("cross-encoder/nli-deberta-v3-small")
print(config.id2label)
