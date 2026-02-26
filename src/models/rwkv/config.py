class RWKVConfig:
    def __init__(self, n_embd=768, n_layer=12):
        self.model_type = "rwkv"
        self.tie_word_embeddings = False
        self.n_embd = n_embd
        self.n_layer = n_layer
        self._name_or_path = "rwkv7-g1d-0.1b"

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)