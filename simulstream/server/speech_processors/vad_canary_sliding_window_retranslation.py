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

from typing import List

from simulstream.server.speech_processors.canary_sliding_window_retranslation import \
    CanarySlidingWindowRetranslator
from simulstream.server.speech_processors.vad_parent import VADParentSpeechProcessor


class VADCanarySlidingWindowRetranslator(VADParentSpeechProcessor):
    """
    Perform Sliding Window Retranslation after VAD speech filtering with Canary.
    """

    speech_processor_class = CanarySlidingWindowRetranslator

    def _tokens_to_string(self, tokens: List[str]) -> str:
        return self.speech_processor._tokens_to_string(tokens)
