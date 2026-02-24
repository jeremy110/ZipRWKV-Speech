from hyperpyyaml import load_hyperpyyaml
from lhotse.dataset import SimpleCutSampler, DynamicBucketingSampler
from torch.utils.data import DataLoader

def test_dataset():

    config_file = "./tests/yaml/test_dataset.yaml"
    with open(config_file) as fin:
        modules = load_hyperpyyaml(fin)

    dataset = modules["train_ds"]
    tokenizer = modules["tokenizer"]

    sampler = DynamicBucketingSampler(
        dataset.cuts_data, 
        max_duration = 100.0, 
        shuffle_buffer_size=1000,
        num_buckets=20,
        shuffle = True
    )
    
    train_dl = DataLoader(
        dataset,
        sampler = sampler,
        batch_size = None,

    ) 
        
    lang_counts = {"en_en": 0, "zh_zh": 0}
    # https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/03-combining-datasets.ipynb#scrollTo=3_rv26K9BuE3

    for i, batch in enumerate(train_dl):
        # print(f"\n--- test Batch {i+1} ---")
        features, feature_lens, asr_texts, ast_texts, languages = batch
        # print(f"features shape (B, T, F): {features.shape} {feature_lens} {languages}")
        # print(f"asr_texts: {asr_texts}")

        ids = [tokenizer.encode(asr_text) for asr_text in asr_texts]
        text = [tokenizer.decode(id) for id in ids]

        print(asr_texts, ids, text)
        assert asr_texts == text

        for lang in languages:
            lang_counts[lang] += 1

        print(lang_counts)

        if i == 2:
            break

    total = sum(lang_counts.values())
    print({k: v/total for k, v in lang_counts.items()})


if __name__ == "__main__":
    test_dataset()