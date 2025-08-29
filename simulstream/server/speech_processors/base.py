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

from abc import abstractmethod
from types import SimpleNamespace
from typing import List, Union, Dict

import numpy as np
import torch

from simulstream.server.speech_processors import SpeechProcessor, IncrementalOutput


class BaseSpeechProcessor(SpeechProcessor):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.tgt_lang_tag = None
        self.src_lang_tag = None
        self.audio_history = None
        self.text_history = None

    @abstractmethod
    def _preprocess(self, waveform: np.float32) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        ...

    @abstractmethod
    def _update_speech_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        ...

    @abstractmethod
    def _update_text_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        ...

    @abstractmethod
    def _generate(self, speech: Union[Dict[str, torch.Tensor], torch.Tensor]) -> List[str]:
        ...

    @abstractmethod
    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        ...

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        speech = self._preprocess(waveform)
        generated_tokens = self._generate(speech)
        new_output = self._build_incremental_outputs(generated_tokens)
        self._update_speech_history(speech, generated_tokens, new_output)
        self._update_text_history(speech, generated_tokens, new_output)
        return new_output

    def clear(self) -> None:
        self.text_history = None
        self.audio_history = None
        self.src_lang_tag = None
        self.tgt_lang_tag = None
