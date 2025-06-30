# استيراد المكتبات
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize

try:
    word_tokenize("اختبار")
except LookupError:
    nltk.download('punkt')

def load_models():
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    return model, tokenizer

model, tokenizer = load_models()

def tokinze_text(text):
    tokens = word_tokenize(text, preserve_line=True)
    filtered_tokens = [token for token in tokens if token.isalpha()]
    return filtered_tokens

def text_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def sentiment_analysis(text):
    stored_db_words = {
        "احنا": "احنا", "تيم": "تيم", "من": "من", "كلية": "كلية", "حاسبات": "حاسبات",
        "معلومات": "معلومات", "جامعة": "جامعة", "المنصورة": "المنصورة", "نستخدم": "نستخدم",
        "التكنولوجيا": "التكنولوجيا", "لحل": "لحل", "المشاكل": "المشاكل", "تحسين": "تحسين",
        "المجتمع": "المجتمع", "تخيل": "تخيل", "إنك": "إنك", "بتتفرج": "بتتفرج", "على": "على",
        "فيديو": "فيديو", "بلا": "بلا", "صوت": "صوت", "وكأنك": "وكأنك", "تعيش": "تعيش",
        "معاناة": "معاناة", "الصم": "الصم", "البكم": "البكم", "يوميًا": "يوميًا", "وسط": "وسط",
        "الناس": "الناس", "فاهم": "فاهم", "كلامهم": "كلامهم", "ده": "بالضبط", "بيعيشه": "بيعيشه",
        "التواصل": "التواصل", "حق": "حق", "للجميع": "للجميع", "بناخد": "بناخد", "خطوة": "خطوة",
        "حقيقية": "حقيقية", "دمج": "دمج",
    }

    stored_words_embeddings = {
        word: text_embedding(word) for word in stored_db_words.keys()
    }

    tokens = tokinze_text(text)

    max_words = []
    corresponding_sign_ids = []

    for token in tokens:
        token_emb = text_embedding(token)
        word_similarities = {
            word: cosine_similarity(token_emb.unsqueeze(0), emb.unsqueeze(0)).item()
            for word, emb in stored_words_embeddings.items()
        }

        max_word = max(word_similarities, key=word_similarities.get)
        max_words.append(max_word)
        corresponding_sign_ids.append(stored_db_words[max_word])

    return max_words, corresponding_sign_ids
