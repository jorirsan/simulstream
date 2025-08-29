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

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import numpy as np


CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16_000


@dataclass
class IncrementalOutput:
    new_tokens: List[str]
    new_string: str
    deleted_tokens: List[str]
    deleted_string: str

    def strings_to_json(self) -> str:
        return json.dumps({"new": self.new_string, "deleted": self.deleted_string})


class SpeechProcessor(ABC):

    def __init__(self, config: SimpleNamespace):
        self.config = config

    @classmethod
    @abstractmethod
    def load_model(cls, config: SimpleNamespace):
        ...

    @abstractmethod
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        ...

    @abstractmethod
    def set_source_language(self, language: str) -> None:
        ...

    @abstractmethod
    def set_target_language(self, language: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


def build_speech_processor(speech_processor_config: SimpleNamespace) -> SpeechProcessor:
    module_path, class_name = speech_processor_config.type.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert issubclass(cls, SpeechProcessor), \
        f"{speech_processor_config} must be a subclass of BaseSpeechProcessor"
    cls.load_model(speech_processor_config)
    return cls(speech_processor_config)
