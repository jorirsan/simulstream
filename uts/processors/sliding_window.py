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

import unittest
from types import SimpleNamespace
from typing import List

from simulstream.server.speech_processors.sliding_window_retranslation import \
    HFSlidingWindowRetranslator


class MockedHFSlidingWindowRetranslator(HFSlidingWindowRetranslator):
    def _tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens).replace("▁", " ")


class SlidingWindowTestCase(unittest.TestCase):
    def test_get_ending_tokens_for_string(self):
        searched_string = \
            ' New York, sono il capo dello sviluppo per un non-profit chiamato Robin Hood.'
        tokens = [
            '▁In', '▁New', '▁York', ',', '▁sono', '▁il', '▁capo', '▁dello', '▁sviluppo', '▁per',
            '▁un', '▁non', '-', 'pro', 'fit', '▁chiama', 'to', '▁Robin', '▁Ho', 'od', '.'
        ]
        config = SimpleNamespace()
        config.window_len = 12
        processor = MockedHFSlidingWindowRetranslator(config)
        found_tokens = processor.get_ending_tokens_for_string(searched_string, tokens)
        self.assertEqual(found_tokens, [
            '▁New', '▁York', ',', '▁sono', '▁il', '▁capo', '▁dello', '▁sviluppo', '▁per',
            '▁un', '▁non', '-', 'pro', 'fit', '▁chiama', 'to', '▁Robin', '▁Ho', 'od', '.'
        ])


if __name__ == '__main__':
    unittest.main()
