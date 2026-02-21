
import torch

from lhotse import CutSet
from lhotse.utils import ifnone
from lhotse.utils import supervision_to_frames

from dataset.nemo_adapters import LazyNeMoLoader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            manifest_path,
            audio_extractor,
            audio_augmentor = None,
            feature_augmentor = None,
            sample_rate = 16000,
            max_length = 30,
            min_length = 0.5,
            use_repeat = False,
            weights = None,
        ):
        super().__init__()

        if isinstance(manifest_path, list):
            cutsets = []
            for path in manifest_path:
                cutsets.append(CutSet(LazyNeMoLoader(path, sample_rate)))
            cutsets = [cs.repeat() for cs in cutsets]
            cuts_data = CutSet.mux(*cutsets, weights = weights, seed = 42)
            use_repeat = False
        elif isinstance(manifest_path, str):
            loader = LazyNeMoLoader(manifest_path, sample_rate)
            cuts_data = CutSet(loader)
        else:
            assert f"manifest_path not support this type: {type(manifest_path)}"

        def filter_fn(cut):
            return min_length <= cut.duration <= max_length
        

        self.cuts_data = cuts_data.filter(filter_fn)
        if use_repeat:
            self.cuts_data = self.cuts_data.repeat()

        self.sample_rate = sample_rate
        self.audio_extractor = audio_extractor
        self.audio_augmentor = ifnone(audio_augmentor, [])
        self.feature_augmentor = ifnone(feature_augmentor, [])
    
        self.compute_feature_lens = lambda *x: supervision_to_frames(*x)[1]

    def __getitem__(self, cuts: CutSet):
        '''
        
            Returns:
                feature: (B, T, F)
                feature_lens: (B)
                texts
                prompts
          
        '''
        # 1. resample to target sample rate (16kHz)
        cuts = CutSet.from_cuts([c.resample(self.sample_rate) for c in cuts])

        # 2. speed or volume aug or add noise
        for tnfm in self.audio_augmentor:
            cuts = tnfm(cuts)

        # 3. convert to Fbank
        feature, _ = self.audio_extractor(cuts)

        # 4. SpecAugment
        for tnfm in self.feature_augmentor:
            feature = tnfm(feature)

        # 5. compute feature lens for attention 
        feature_lens = [self.compute_feature_lens(supervision, cut.frame_shift if cut.frame_shift else self.audio_extractor.extractor.frame_shift,
                                                  self.sample_rate) for _, cut in enumerate(cuts) for supervision in cut.supervisions]
        feature_lens = torch.IntTensor(feature_lens)

        # 6. process texts and tags
        asr_texts = [sup.text for c in cuts for sup in c.supervisions]
        languages = [f"{sup.language}_{sup.custom['tgt_lang']}" for c in cuts for sup in c.supervisions]
        ast_texts = [sup.custom['ast_text'] for c in cuts for sup in c.supervisions]
        

        return feature, feature_lens, asr_texts, ast_texts, languages

