import unittest

from eval.run_mbpp import (
    build_mbpp_instruction,
    build_prompt,
    extract_code,
    run_candidate_tests,
)


class MbppEvalTests(unittest.TestCase):
    def test_build_prompt_uses_mbpp_prompt(self):
        prompt = build_prompt("Write a function to add two numbers.")

        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("Write a function to add two numbers.", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_build_mbpp_instruction_includes_tests_for_signature(self):
        instruction = build_mbpp_instruction(
            problem_text="Write a function to remove first and last occurrence of a given character.",
            test_list=[
                "assert remove_Occ('hello','l') == 'heo'",
                "assert remove_Occ('abcda','a') == 'bcd'",
            ],
        )

        self.assertIn("remove first and last occurrence", instruction)
        self.assertIn("assert remove_Occ('hello','l') == 'heo'", instruction)
        self.assertIn("remove_Occ", instruction)

    def test_extract_code_removes_markdown_fence(self):
        generated = "```python\ndef add(a, b):\n    return a + b\n```"

        self.assertEqual(extract_code(generated), "def add(a, b):\n    return a + b")

    def test_run_candidate_tests_passes_correct_solution(self):
        result = run_candidate_tests(
            candidate_code="def add(a, b):\n    return a + b",
            test_imports=[],
            test_list=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
            timeout_seconds=2,
        )

        self.assertTrue(result.passed)
        self.assertIsNone(result.error)

    def test_run_candidate_tests_fails_wrong_solution(self):
        result = run_candidate_tests(
            candidate_code="def add(a, b):\n    return a - b",
            test_imports=[],
            test_list=["assert add(1, 2) == 3"],
            timeout_seconds=2,
        )

        self.assertFalse(result.passed)
        self.assertIn("AssertionError", result.error)


if __name__ == "__main__":
    unittest.main()
