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
        window.
        """
        new_string = self.processor.tokenizer.convert_tokens_to_string(generated_tokens)
        if self.text_history is None or len(self.text_history) == 0:
            self.text_history = generated_tokens
            return IncrementalOutput(
                new_tokens=generated_tokens,
                new_string=new_string,
                deleted_tokens=[],
                deleted_string=""
            )
        previous_string = self.processor.tokenizer.convert_tokens_to_string(self.text_history)
        seq_matcher = SequenceMatcher(None, previous_string, new_string)
        longest_match = seq_matcher.find_longest_match()
        if longest_match.size >= self.matching_threshold * len(new_string):
            new_string = new_string[longest_match.b + longest_match.size:]
            deleted_string = previous_string[longest_match.a + longest_match.size:]
            if new_string == '':
                new_tokens = []
            else:
                new_tokens = self.get_ending_tokens_for_string(new_string, generated_tokens)
            if deleted_string == '':
                deleted_tokens = []
            else:
                deleted_tokens = self.get_ending_tokens_for_string(
                    deleted_string, self.text_history)
            # we take the matching part and the last part of the generated string as part of
            # the history
            self.text_history = self.get_ending_tokens_for_string(
                new_string[longest_match.b:], generated_tokens)
        else:
            deleted_tokens = []
            deleted_string = ""
            new_tokens = generated_tokens
            self.text_history = generated_tokens
        return IncrementalOutput(
            new_tokens=new_tokens,
            new_string=new_string,
            deleted_tokens=deleted_tokens,
            deleted_string=deleted_string,
        )

    def get_ending_tokens_for_string(self, string: str, tokens: List[str]) -> List[str]:
        i = 1
        while i < len(tokens):
            ending_tokens_string = self.processor.tokenizer.convert_tokens_to_string(tokens[-i:])
            if not string.endswith(ending_tokens_string):
                return tokens[-i + 1:]
            i += 1
        return tokens

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
