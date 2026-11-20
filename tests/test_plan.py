import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_analyzer.plan import parse_folders, read_plan, validate_folders, write_plan


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = os.path.join(self.tmp.name, "src")
        os.mkdir(self.source)
        for name in ("lua.txt", "mars.txt", "id.pdf"):
            open(os.path.join(self.source, name), "w").close()
        self.plan = os.path.join(self.tmp.name, "plan.txt")

    def test_round_trip_lists_weakest_first(self):
        folders = {"Solar_system": ["mars.txt", "lua.txt"], "Misc": ["id.pdf"]}
        write_plan(self.plan, folders, self.source, self.source,
                   scores={"mars.txt": 0.83, "lua.txt": 0.41, "id.pdf": 0.2}, sensitive={"id.pdf": ["cpf"]})
        with open(self.plan, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("0.41\tlua.txt\n0.83\tmars.txt", text)
        self.assertIn("# sensitive: cpf\n0.20\tid.pdf", text)

        read, source, target = read_plan(self.plan)
        self.assertEqual(read, {"Solar_system": ["lua.txt", "mars.txt"], "Misc": ["id.pdf"]})
        self.assertEqual((source, target), (os.path.abspath(self.source),) * 2)

    def test_hand_edits_are_read_back(self):
        """Rename a folder, move a line, delete a line, drop the scores."""
        with open(self.plan, "w", encoding="utf-8") as f:
            f.write(f"# organize-files plan\tsource={self.source}\ttarget={self.source}\n"
                    "[Astronomy]\nmars.txt\n0.41\tlua.txt\n\n[Empty now]\n")
        folders, _, _ = read_plan(self.plan)
        self.assertEqual(folders, {"Astronomy": ["mars.txt", "lua.txt"]})

    def test_not_a_plan(self):
        with open(self.plan, "w") as f:
            f.write("[Folder]\nlua.txt\n")
        with self.assertRaises(ValueError):
            read_plan(self.plan)

    def test_validation_rejects_unsafe_or_wrong_entries(self):
        validate_folders({"Space/Planets": ["mars.txt"], "Moon": ["lua.txt"]}, self.source)  # fine
        for folders in (
            {"../outside": ["lua.txt"]},
            {"/abs": ["lua.txt"]},
            {"a\\b": ["lua.txt"]},
            {"C:x": ["lua.txt"]},
            {" ": ["lua.txt"]},
            {"Ok": ["../etc/passwd"]},
            {"Ok": ["missing.txt"]},
            {"A": ["lua.txt"], "B": ["lua.txt"]},
        ):
            with self.subTest(folders=folders), self.assertRaises(ValueError):
                validate_folders(folders, self.source)

    def test_parse_folders(self):
        self.assertIsNone(parse_folders(""))
        self.assertIsNone(parse_folders(None))
        self.assertEqual(parse_folders(" 5 "), 5)
        self.assertEqual(parse_folders("Finanças, Receitas, Finanças,"), ["Finanças", "Receitas"])
        for bad in ("0", "../fora", "C:\\x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_folders(bad)


if __name__ == "__main__":
    unittest.main()
