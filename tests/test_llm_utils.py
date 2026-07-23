import unittest
from unittest.mock import Mock

from llm_utils import get_chat_model_candidates, run_chat_completion


class LlmUtilsTests(unittest.TestCase):
    def test_get_chat_model_candidates_uses_configured_model_first(self):
        candidates = get_chat_model_candidates("custom-model")
        self.assertEqual(candidates[0], "custom-model")
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", candidates)

    def test_run_chat_completion_falls_back_when_model_is_unsupported(self):
        client = Mock()
        unsupported_error = Exception("The requested model 'bad-model' is not supported by any provider")
        success_response = object()
        client.chat_completion.side_effect = [unsupported_error, success_response]

        response, model = run_chat_completion(client, model="bad-model", messages=[{"role": "user", "content": "hi"}])

        self.assertIs(response, success_response)
        self.assertEqual(model, "Qwen/Qwen2.5-3B-Instruct")


if __name__ == "__main__":
    unittest.main()
