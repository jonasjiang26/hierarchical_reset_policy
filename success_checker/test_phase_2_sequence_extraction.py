from __future__ import annotations

import unittest

from success_checker.phase_2_checker import Phase2SuccessChecker


EXPECTED_STRAWBERRY_RESET_SEQUENCE = [
    "open the lower drawer.",
    "put the strawberry from plate back in drawer.",
    "close the lower drawer.",
]


class ResetSequenceExtractionTest(unittest.TestCase):
    def test_ignores_think_block_and_uses_final_bracketed_sequence(self) -> None:
        llm_response = """
<think>
So the reset sequence is: [open the lower drawer. put the strawberry from plate back in drawer. close the lower drawer.]
</think>

task succeeded
[open the lower drawer. put the strawberry from plate back in drawer. close the lower drawer.]
"""

        self.assertEqual(
            Phase2SuccessChecker._extract_reset_sequence(llm_response),
            EXPECTED_STRAWBERRY_RESET_SEQUENCE,
        )

    def test_uses_first_answer_bracket_before_echoed_publish_logs(self) -> None:
        llm_response = """
task succeeded
[open the lower drawer. put the strawberry from plate back in drawer. close the lower drawer.]
Published reset sequence: ['open the lower drawer.', 'put the strawberry from plate back in drawer.', 'close the lower drawer.', 'open the lower drawer.', 'put the strawberry from plate back in drawer.', 'close the lower drawer.']
"""

        self.assertEqual(
            Phase2SuccessChecker._extract_reset_sequence(llm_response),
            EXPECTED_STRAWBERRY_RESET_SEQUENCE,
        )

    def test_collapses_repeated_plain_text_sequence(self) -> None:
        llm_response = """
task succeeded
open the lower drawer.
put the strawberry from plate back in drawer.
close the lower drawer.
open the lower drawer.
put the strawberry from plate back in drawer.
close the lower drawer.
"""

        self.assertEqual(
            Phase2SuccessChecker._extract_reset_sequence(llm_response),
            EXPECTED_STRAWBERRY_RESET_SEQUENCE,
        )


if __name__ == "__main__":
    unittest.main()
