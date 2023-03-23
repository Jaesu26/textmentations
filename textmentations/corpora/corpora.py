from .raw_corpora import RAW_STOPWORDS

# TODO: final로 선언
STOPWORDS = set(RAW_STOPWORDS.get("stopwords", []))
