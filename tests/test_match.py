import unittest

from match import to_label


LABELS = ["Applicable Laws", "Governing Laws", "No Waivers", "Waivers"]


class ToLabelTests(unittest.TestCase):
    def test_exact_label(self):
        self.assertEqual(to_label("Waivers", LABELS), "Waivers")

    def test_label_wrapped_in_sentence(self):
        self.assertEqual(to_label("The answer is: Applicable Laws.", LABELS), "Applicable Laws")

    def test_minor_variation(self):
        self.assertEqual(to_label("governing law", LABELS), "Governing Laws")

    def test_empty_output_fails_closed(self):
        self.assertIsNone(to_label("", LABELS))
        self.assertIsNone(to_label(None, LABELS))

    def test_unrelated_output_fails_closed(self):
        self.assertIsNone(to_label("I cannot classify this request", LABELS))
        self.assertIsNone(to_label("a", LABELS))


if __name__ == "__main__":
    unittest.main()
