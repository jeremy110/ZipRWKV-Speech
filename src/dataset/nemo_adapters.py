from lhotse.lazy import LazyJsonlIterator
from lhotse import Recording, SupervisionSegment

class LazyNeMoLoader:
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
    ):
        self.path = manifest_path
        self.sample_rate = sample_rate
        self.source = LazyJsonlIterator(manifest_path)

    def __iter__(self):
        for data in self.source:
            # 1. According to the columns of nemo manifest
            audio_path = data["audio_filepath"]
            duration = data["duration"]
            offset = data.get("offset", 0.0)
            asr_text = data.get("asr_text", "")
            ast_text = data.get("ast_text", "")
            src_lang = data.get("source_lang", "zh")
            tgt_lang = data.get("target_lang", src_lang)

            # 2. create lhotse cut
            cut = self._create_cut(audio_path, offset, duration)

            # 3. create Supervision (ASR/AST tags)
            cut.supervisions.append(
                SupervisionSegment(
                    id = f"sup_{cut.id}",
                    recording_id = cut.recording_id,
                    start = 0,
                    duration = cut.duration,
                    text = asr_text,
                    language = src_lang,
                    custom = {
                        "tgt_lang": tgt_lang,
                        "ast_text": ast_text,
                    }
                )
            )

            yield cut

    def _create_cut(
        self,
        audio_path: str,
        offset: float,
        duration: float,
    ):
        recording = Recording.from_file(audio_path)
        cut = recording.to_cut()
        if offset > 0 or duration < recording.duration:
            cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
            cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"

        return cut

    def __len__(self):
        return len(self.source)