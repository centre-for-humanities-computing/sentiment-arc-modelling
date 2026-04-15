import numpy as np
from spacy.lang.en import English
from tqdm import tqdm
from turftopic import ConceptVectorProjection
from turftopic.late import LateSentenceTransformer, LateWrapper

SEED_PHRASES = [
    (
        "monetary_sentiment",
        (
            # HAWKISH
            [
                "Inflation remains too high and requires further policy tightening.",
                "Rising inflationary pressures call for higher interest rates.",
                "Strong wage growth poses upside risks to inflation.",
                "Financial conditions remain loose given current inflation dynamics.",
                "Interest rates will need to increase further to ensure price stability.",
                "Persistent inflation requires a firm and timely policy response.",
                "Elevated inflation expectations warrant decisive action.",
                "Price pressures are broadening and justify higher rates.",
                "Monetary tightening is necessary to prevent overheating.",
                "We are prepared to raise interest rates if inflation accelerates.",
                "Further rate increases are likely to be appropriate in the coming meetings.",
            ],
            # DOVISH
            [
                "Economic conditions warrant maintaining a low policy rate.",
                "We will continue asset purchases to support favorable financing conditions.",
                "Policy rates will remain at current or lower levels for an extended period.",
                "Accommodative monetary policy remains necessary to support the recovery.",
                "Weak demand and subdued inflation call for supportive policy measures.",
                "Liquidity provisions will continue to safeguard smooth market functioning.",
                "We will reinvest maturing securities for as long as needed.",
                "Forward guidance indicates continued policy accommodation.",
                "Low interest rates remain appropriate while inflation is below target.",
                "Asset purchases help maintain favorable financial conditions.",
                "The current stance supports lending and economic activity.",
                "Liquidity operations will continue to ease financing conditions.",
                "Maintaining accommodative policy will help restore inflation to target.",
            ],
        ),
    ),
    (
        "economic_sentiment",
        (
            # POSITIVE
            [
                "The economy is expanding at a solid pace with strong job growth.",
                "Consumer demand remains strong and supports continued growth.",
                "Labor-market conditions are strong and improving.",
                "Business investment continues to strengthen.",
                "Economic momentum has increased across many sectors.",
                "Job gains remain robust and unemployment is low.",
                "Financial conditions remain supportive of growth.",
                "Economic indicators point to a sustained expansion.",
                "Productivity gains continue to support economic growth.",
                "Confidence indicators suggest improving economic outlook.",
                "Household spending is strong and continues to improve.",
                "Exports and external demand remain supportive.",
                "Economic activity continues to progress at a solid pace.",
                "Broad-based growth supports continued recovery.",
            ],
            # NEGATIVE
            [
                "Weak demand and tight credit conditions are restraining growth.",
                "Household spending remains constrained by low income growth.",
                "Financial-market stress is weighing on economic activity.",
                "Job losses and declining wealth are weakening sentiment.",
                "Rising energy prices are reducing purchasing power.",
                "Industrial production and exports are weakening.",
                "Growth remains fragile and uneven across sectors.",
                "Economic activity is slowing and confidence is deteriorating.",
                "Credit conditions remain tight and continue to constrain spending.",
                "Inflation remains below target and reflects economic weakness.",
                "Weak investment and productivity are dampening outlook.",
                "Economic recovery remains slow and vulnerable.",
                "Business surveys signal worsening economic conditions.",
            ],
        ),
    ),
    (
        "uncertainty",
        (
            # HIGH UNCERTAINTY
            [
                "The outlook remains highly uncertain.",
                "The recovery process is likely to be uneven and subject to high uncertainty.",
                "Economic growth is expected to remain uneven in an environment of uncertainty.",
                "The outlook is subject to particularly high uncertainty and intensified downside risks.",
                "The near-term economic outlook remains clouded by uncertainty.",
                "The speed and scale of the recovery remain highly uncertain.",
                "Pandemic-related uncertainty is likely to dampen the recovery in consumption, investment, and labour markets.",
                "Geopolitical and financial uncertainty continue to weigh on the growth outlook.",
                "Heightened uncertainty makes policy flexibility especially important.",
                "Substantial uncertainty surrounds the timing and pace of the improvement.",
                "The course of the virus and the outlook for the economy remain highly uncertain.",
                "Significant uncertainty surrounds the strength of final demand.",
                "Business investment remains a major source of uncertainty for the overall outlook.",
                "The future course of the economy is subject to a marked degree of uncertainty.",
                "Elevated uncertainty around the economic outlook justifies maintaining the policy stance.",
            ],
            # LOW UNCERTAINTY
            [
                "Available information is broadly in line with the baseline scenario.",
                "Inflation is expected to remain moderate and consistent with price stability.",
                "Risks to the inflation outlook are broadly balanced.",
                "Inflation expectations remain firmly anchored.",
                "Current inflation developments are in line with previous expectations.",
                "Wage and price developments remain subdued and consistent with price stability.",
                "The Governing Council stands ready to act to preserve price stability and keep expectations anchored.",
                "Inflation is expected to stabilize around the Committee's 2 percent objective over the medium term.",
                "Longer-term inflation expectations remain well anchored.",
                "Incoming data are broadly in line with staff expectations.",
                "The current policy stance remains appropriate given the baseline outlook.",
                "Resource slack is expected to keep inflation contained.",
                "Financial markets largely anticipate no change in the policy rate.",
            ],
        ),
    ),
]


class SentenceSeparatedEncoder:
    def __init__(self, encoder: LateSentenceTransformer):
        self.encoder = encoder

    def encode(self, sentences, *args, **kwargs):
        return self.encoder.encode(sentences, *args, **kwargs)

    def encode_tokens(self, texts, batch_size=32, show_progress_bar=True) -> tuple:
        nlp = English()
        nlp.add_pipe("sentencizer")
        embeddings = []
        offsets = []
        for text in tqdm(texts, desc="Producing embeddings for all sentences..."):
            doc = nlp(text)
            sentences = doc.sents
            doc_embeddings = []
            doc_offsets = []
            for sentence in sentences:
                sentence_start = doc[sentence.start].idx
                sent_embeddings, sent_offsets = self.encoder.encode_tokens(
                    [sentence.text], batch_size=batch_size, show_progress_bar=False
                )
                sent_embeddings = sent_embeddings[0]
                sent_offsets = sent_offsets[0]
                _offsets = []
                for start, end in sent_offsets:
                    if (start == 0) and (end == 0):
                        _offsets.append((0, 0))
                    else:
                        _offsets.append((sentence_start + start, sentence_start + end))
                doc_embeddings.extend(sent_embeddings)
                doc_offsets.extend(_offsets)
            embeddings.append(np.stack(doc_embeddings))
            offsets.append(doc_offsets)
        return embeddings, offsets


def load_model(batch_size: int = 8, separate_sentences: bool = False) -> LateWrapper:
    """Construct and load CVP model"""
    encoder = LateSentenceTransformer(
        "nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True,
        model_kwargs={
            "attn_implementation": "eager",  # Or "flash_attention_2"
            "torch_dtype": "bfloat16",
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    if separate_sentences:
        encoder = SentenceSeparatedEncoder(encoder)
    model = LateWrapper(
        ConceptVectorProjection(seeds=SEED_PHRASES, encoder=encoder),
        batch_size=batch_size,
    )
    return model
