# Hugging Face LLM
from transformers import pipeline

translation_pipeline = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    tokenizer="facebook/nllb-200-distilled-600M"
)

def coll2formal_translation(processed_text, pipeline=translation_pipeline):
    translated_text = pipeline(processed_text, src_lang="arz_Arab", tgt_lang="arb_Arab")
    return translated_text[0]["translation_text"]
