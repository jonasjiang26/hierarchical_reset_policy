from __future__ import annotations

import unittest

from success_checker.phase_2_checker import Phase2SuccessChecker


EXPECTED_PAN_RESET_SEQUENCE = [
    "put carrot back in sink",
    "put pan back in place from stove",
]


class ResetSequenceExtractionTest(unittest.TestCase):
    def test_ignores_think_block_and_uses_final_bracketed_sequence(self) -> None:
        llm_response = """
<think>
So the reset sequence is: [put carrot back in sink][put pan back in place from stove]
</think>

task failed
[put carrot back in sink]
[put pan back in place from stove]
"""

        self.assertEqual(
            Phase2SuccessChecker._extract_reset_sequence(llm_response),
            EXPECTED_PAN_RESET_SEQUENCE,
        )

    def test_uses_first_answer_bracket_before_echoed_publish_logs(self) -> None:
        llm_response = """
task failed
[put carrot back in sink]
[put pan back in place from stove]
Published reset sequence: ['put carrot back in sink', 'put pan back in place from stove', 'put carrot back in sink', 'put pan back in place from stove']
"""

        self.assertEqual(
            Phase2SuccessChecker._extract_reset_sequence(llm_response),
            EXPECTED_PAN_RESET_SEQUENCE,
        )

    def test_collapses_repeated_plain_text_sequence(self) -> None:
        llm_response = """
task failed
put carrot back in sink
put pan back in place from stove
put carrot back in sink
put pan back in place from stove
"""

        self.assertEqual(
            Phase2SuccessChecker._extract_reset_sequence(llm_response),
            EXPECTED_PAN_RESET_SEQUENCE,
        )


if __name__ == "__main__":
    unittest.main()
