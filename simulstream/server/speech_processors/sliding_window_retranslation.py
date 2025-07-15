# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base import BaseSpeechProcessor, IncrementalOutput


class HFSlidingWindowRetranslator(BaseSpeechProcessor):

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "model") or cls.model is None:
            lang_tags = None
            if hasattr(config, "supported_langs") and config.supported_langs is not None:
                lang_tags = [
                    config.lang_tag_template.format(lang) for lang in config.supported_langs]
            cls.processor = AutoProcessor.from_pretrained(
                config.hf_model_name,
                additional_special_tokens=lang_tags)
            cls.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                config.hf_model_name, trust_remote_code=True)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.model.to(cls.device)

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.window_len = self.config.window_len * SAMPLE_RATE
        self.matching_threshold = getattr(self.config, "matching_threshold", 0.1)

    def _generate(self, speech: torch.Tensor) -> List[str]:
        extra_kwargs = {}
        if self.lang_tag_id is not None:
            extra_kwargs["forced_bos_token_id"] = self.lang_tag_id
        generated_ids = self.model.generate(speech, **extra_kwargs)[0]
        return self.processor.tokenizer.convert_ids_to_tokens(
            generated_ids, skip_special_tokens=True)

    def _tokens_to_string(self, tokens: List[str]) -> str:
        return self.processor.tokenizer.convert_tokens_to_string(tokens)

    def _preprocess(self, waveform: np.float32) -> torch.Tensor:
        """
        Extracts the filter-bank features from the input waveform and appends them to the audio
        history. Returns the concatenated audio history and new frames, taking the last
        `self.window_len` frames, and returns it after storing it in the audio history.
        """
        if self.audio_history is not None:
            waveform = np.concatenate((self.audio_history, waveform))
        new_speech_len = len(waveform)
        if new_speech_len > self.window_len:
            waveform = waveform[-self.window_len:]
        self.audio_history = waveform
        new_speech = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt")["input_features"]
        return new_speech.to(self.device)

    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        """
        Deduplicates the output stream of overlapping windows using the algorithm introduced in
        `S. Sen, et al. 2025. "Simultaneous Translation for Unsegmented Input:
        A Sliding Window Approach" <https://arxiv.org/pdf/2210.09754>`_

        This algorithm is based on the longest matching substring between the current and previous
        window. We use tokens instead of string to match, though, as we have empirically observed
        that tokenization is mostly consistent across generations of the same word.
        """
        if self.text_history is None or len(self.text_history) == 0:
            self.text_history = generated_tokens
            generated_string = self._tokens_to_string(generated_tokens)
            return IncrementalOutput(
                new_tokens=generated_tokens,
                new_string=generated_string,
                deleted_tokens=[],
                deleted_string=""
            )
        seq_matcher = SequenceMatcher(None, self.text_history, generated_tokens, autojunk=False)
        longest_match = seq_matcher.find_longest_match()
        if longest_match.size >= self.matching_threshold * len(generated_tokens):
            new_tokens = generated_tokens[longest_match.b + longest_match.size:]
            deleted_tokens = self.text_history[longest_match.a + longest_match.size:]
            new_string = self._tokens_to_string(new_tokens)
            deleted_string = self._tokens_to_string(deleted_tokens)
            # we take the matching part and the last part of the generated string as part of
            # the history. Then we take from the history the tokens corresponding to the amount
            # generated in this step, to ensure we have a sufficiently wide window
            matching_and_last_tokens = generated_tokens[longest_match.b:]
            initial_discarded_tokens = len(generated_tokens) - len(matching_and_last_tokens)
            history_tokens_discarded = self.text_history[longest_match.a:]
            history_initial_tokens = len(self.text_history) - len(history_tokens_discarded)
            new_history_initial_tokens = self.text_history[
                max(history_initial_tokens - initial_discarded_tokens, 0):history_initial_tokens]
            self.text_history = new_history_initial_tokens + matching_and_last_tokens
        else:
            deleted_tokens = []
            deleted_string = ""
            new_tokens = generated_tokens
            new_string = self._tokens_to_string(generated_tokens)
            self.text_history = generated_tokens
        return IncrementalOutput(
            new_tokens=new_tokens,
            new_string=new_string,
            deleted_tokens=deleted_tokens,
            deleted_string=deleted_string,
        )

    def set_language(self, language: str) -> None:
        lang_tag_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.config.lang_tag_template.format(language))
        self.lang_tag_id = torch.tensor(lang_tag_id, dtype=torch.int, device=self.device)

    def _update_speech_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        pass

    def _update_text_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        pass

    def clear(self) -> None:
        self.text_history = None
        self.audio_history = None
        self.lang_tag_id = None
