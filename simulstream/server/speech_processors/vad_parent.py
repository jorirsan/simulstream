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
from typing import List

import numpy as np
from silero_vad import load_silero_vad, VADIterator

from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor
from simulstream.server.speech_processors.base import IncrementalOutput


class VADParentSpeechProcessor(SpeechProcessor):

    @classmethod
    @property
    @abstractmethod
    def speech_processor_class(cls) -> type[SpeechProcessor]:
        ...

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        super().load_model(config)
        if not hasattr(cls, "vad_model") or cls.vad_model is None:
            cls.vad_model = load_silero_vad()
        cls.speech_processor_class.load_model(config)

    @abstractmethod
    def _tokens_to_string(self, tokens: List[str]) -> str:
        ...

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.speech_processor = self.speech_processor_class(self.config)
        self.min_speech_size = getattr(self.config, "min_speech_size", 1) * SAMPLE_RATE
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=getattr(self.config, "vad_threshold", 0.5),
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=getattr(self.config, "vad_min_silence_duration_ms", 100),
            speech_pad_ms=getattr(self.config, "vad_speech_pad_ms", 30),
        )
        self.residual_prev_audio = None
        self.speech_buffer = None
        self.audio_cursor_position = 0
        self.in_speech = False
        assert SAMPLE_RATE == 16000, \
            "SileroHFSlidingWindowRetranslator supports only 16kHz sampling rate"
        self.window_size_samples = 512  # assuming 16kHz

    def clear(self) -> None:
        super().clear()
        self.residual_prev_audio = None
        self.speech_buffer = None
        self.audio_cursor_position = 0
        self.in_speech = False
        self.vad_iterator.reset_states()
        self.speech_processor.clear()

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        if self.residual_prev_audio is not None:
            waveform = np.concatenate((self.residual_prev_audio, waveform))
            self.residual_prev_audio = None
        # we can have more than one generate if there are multiple speech segments in the current
        # chunk
        outputs = []

        for i in range(0, len(waveform), self.window_size_samples):
            chunk = waveform[i: i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                # process tailing audio with the next waveform chunk
                self.residual_prev_audio = chunk
                break
            speech_dict = self.vad_iterator(chunk, return_seconds=False)
            if speech_dict:
                # if a VAD event happens, update the status accordingly
                assert not ('start' in speech_dict and 'end' in speech_dict)
                if 'start' in speech_dict:
                    assert not self.in_speech, \
                        "Cannot start a new segment when current one is being processed. " \
                        "This means there is a bug in the implementation."
                    chunk_start_position = speech_dict['start'] - self.audio_cursor_position
                    self.speech_buffer = chunk[chunk_start_position:]
                    self.in_speech = True
                if 'end' in speech_dict:
                    assert self.in_speech, \
                        "Cannot end a segment when no current segment is being processed. " \
                        "This means there is a bug in the implementation."
                    chunk_start_position = speech_dict['end'] - self.audio_cursor_position
                    self.speech_buffer = np.concatenate(
                        (self.speech_buffer, chunk[:chunk_start_position]))
                    self.in_speech = False
                    outputs.append(self.speech_processor.process_chunk(self.speech_buffer))
                    self.speech_buffer = None
                    # reset history at the end of a segment
                    self.text_history = None
                    self.audio_history = None
            else:
                # if no VAD event happens, we just ignore the audio if we are outside speech and
                # update the buffer in case we are in speech
                if self.in_speech:
                    self.speech_buffer = np.concatenate((self.speech_buffer, chunk))
            # update cursor position
            self.audio_cursor_position += self.window_size_samples

        if self.in_speech and len(self.speech_buffer) > self.min_speech_size:
            outputs.append(self.speech_processor.process_chunk(self.speech_buffer))

        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) == 0:
            return IncrementalOutput([], "", [], "")
        else:
            return self._merge_incremental_outputs(outputs)

    def _merge_incremental_outputs(self, outputs: List[IncrementalOutput]) -> IncrementalOutput:
        current_output_tokens = outputs[0].new_tokens
        current_output_deleted_tokens = outputs[0].deleted_tokens
        for output in outputs[1:]:
            num_deleted_tokens = len(output.deleted_tokens)
            if num_deleted_tokens > 0:
                if num_deleted_tokens < len(current_output_tokens):
                    assert output.deleted_tokens == current_output_tokens[-num_deleted_tokens:]
                    current_output_tokens = current_output_tokens[:-num_deleted_tokens]
                else:
                    # we are deleting more than it was generated so far, so extra deleted tokens
                    # should be included
                    extra_deleted_tokens = output.deleted_tokens[:-len(current_output_tokens)]
                    current_output_deleted_tokens = \
                        extra_deleted_tokens + current_output_deleted_tokens
                    current_output_tokens = []
            current_output_tokens += output.new_tokens

        return IncrementalOutput(
            current_output_tokens,
            self._tokens_to_string(current_output_tokens),
            current_output_deleted_tokens,
            self._tokens_to_string(current_output_deleted_tokens))

    def set_source_language(self, language: str) -> None:
        self.speech_processor.set_source_language(language)

    def set_target_language(self, language: str) -> None:
        self.speech_processor.set_target_language(language)
