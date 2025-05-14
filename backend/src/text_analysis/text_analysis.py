
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams


def compute_tfidf_scores(texts):
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        avg_tfidf = tfidf_matrix.mean()
        return avg_tfidf
    except ValueError:
        return 0.0


def top_words(texts, top_n=5):
    all_words = " ".join(texts).split()
    counter = Counter(all_words)
    return counter.most_common(top_n)


def distinct_ngrams(texts, n=1):
    ngram_set = set()
    for text in texts:
        tokens = text.split()
        ngram_set.update(ngrams(tokens, n))
    return len(ngram_set)


def analyze_text_statistics(texts):
    lengths = [len(t.split()) for t in texts if t.strip()]
    if not lengths:
        return {
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
            "vocab_size": 0,
            "top_words": [],
            "distinct_unigrams": 0,
            "distinct_bigrams": 0,
            "distinct_trigrams": 0,
            "distinct_1_ratio": 0.0,
            "distinct_2_ratio": 0.0,
            "avg_tfidf": 0.0
        }

    vocab = set(" ".join(texts).split())

    total_tokens = sum(lengths)
    unigram_count = distinct_ngrams(texts, 1)
    bigram_count = distinct_ngrams(texts, 2)

    return {
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "vocab_size": len(vocab),
        "top_words": top_words(texts),
        "distinct_unigrams": unigram_count,
        "distinct_bigrams": bigram_count,
        "distinct_trigrams": distinct_ngrams(texts, 3),
        "distinct_1_ratio": unigram_count / total_tokens if total_tokens else 0.0,
        "distinct_2_ratio": bigram_count / total_tokens if total_tokens else 0.0,
        "avg_tfidf": compute_tfidf_scores(texts)
    }


def compare_datasets(original_texts, synthetic_texts):
    smooth_fn = SmoothingFunction().method1
    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores, jaccard_scores = [], [], [], [], []

    for orig, synth in zip(original_texts, synthetic_texts):
        ref = orig.split()
        hyp = synth.split()

        bleu_1_scores.append(sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smooth_fn))
        bleu_2_scores.append(sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn))
        bleu_3_scores.append(sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn))
        bleu_4_scores.append(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn))

        set_ref = set(ref)
        set_hyp = set(hyp)
        jaccard_scores.append(len(set_ref & set_hyp) / len(set_ref | set_hyp) if set_ref | set_hyp else 0.0)

    return {
        "avg_bleu1": sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0.0,
        "avg_bleu2": sum(bleu_2_scores) / len(bleu_2_scores) if bleu_2_scores else 0.0,
        "avg_bleu3": sum(bleu_3_scores) / len(bleu_3_scores) if bleu_3_scores else 0.0,
        "avg_bleu4": sum(bleu_4_scores) / len(bleu_4_scores) if bleu_4_scores else 0.0,
        "avg_jaccard": sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
    }
