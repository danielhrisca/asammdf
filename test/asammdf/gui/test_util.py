import inspect
import unittest

from asammdf.gui.utils import generate_python_function
from test.asammdf.gui.resources.functions import (
    Function1,
    gray2dec,
    maximum,
    rpm_to_rad_per_second,
)


class TestUtils(unittest.TestCase):
    def test_GeneratePythonFunction_InvalidArgs(self):
        """
        Events:
            - Call function 'generate_python_function' with invalid args type.
        Evaluate:
            - Evaluate return type is tuple
            - Evaluate func is always None. (position 0)
            - Evaluate trace. (position 1)
        """
        # 'definition' argument is not string.
        with self.subTest(f"{self.id()}_0"):
            trace = "The function definition must be a string"

            # Event
            result = generate_python_function(None, None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertEqual(trace, result[1])

        # 'in_globals' argument is not dict.
        with self.subTest(f"{self.id()}_1"):
            trace = "'in_globals' must be a dict"

            # Event
            result = generate_python_function(self.id(), True)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertEqual(trace, result[1])

    def test_GeneratePythonFunction_InconsistentArgs(self):
        """
        Events:
            - Call function 'generate_python_function' with valid args type but inconsistent content.
        Evaluate:
            - Evaluate return type is tuple
            - Evaluate func is always None. (position 0)
            - Evaluate trace. (position 1)
        """
        # Empty definition
        with self.subTest(f"{self.id()}_0"):
            trace = "The function definition must not be empty"

            # Event
            result = generate_python_function("", None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertEqual(trace, result[1])

        # Function Name absent
        with self.subTest(f"{self.id()}_1"):
            trace = "The function name must not be empty"

            # Event
            result = generate_python_function("\t", None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertEqual(trace, result[1])

    def test_GeneratePythonFunction_Exception(self):
        """
        Events:
            - Call function 'generate_python_function' with syntax errors in definition.
        Evaluate:
            - Evaluate return type is tuple
            - Evaluate func is always None. (position 0)
            - Evaluate trace. (position 1)
        """
        # Event
        result = generate_python_function(r"def Function1(t=0):\n    return true", None)

        # Evaluate
        self.assertIsInstance(result, tuple)
        self.assertEqual(None, result[0])
        self.assertIn(
            "SyntaxError: unexpected character after line continuation character",
            result[1],
        )

    def test_GeneratePythonFunction_Args(self):
        """
        Events:
            - Call function 'generate_python_function' without argument 't=0' in definition or on wrong position
        Evaluate:
            - Evaluate return type is tuple
            - Evaluate func is always None. (position 0)
            - Evaluate trace. (position 1)
        """
        with self.subTest(f"{self.id()}_0"):
            trace = 'The last function argument must be "t=0"'

            # Event
            result = generate_python_function("def Function1():\n\treturn 0", None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertIn(trace, result[1])

        with self.subTest(f"{self.id()}_1"):
            trace = 'The last function argument must be "t=0"'

            # Event
            result = generate_python_function("def Function1(t=0, x=0):\n\treturn 0", None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertIn(trace, result[1])

        with self.subTest(f"{self.id()}_2"):
            trace = 'The last function argument must be "t=0"'

            # Event
            result = generate_python_function("def Function1(t=1):\n\treturn 0", None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertIn(trace, result[1])

        with self.subTest(f"{self.id()}_3"):
            trace = 'All the arguments must have default values. The argument "channel" has no default value.'

            # Event
            result = generate_python_function("def Function1(channel, t=0):\n\treturn 0", None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertEqual(None, result[0])
            self.assertIn(trace, result[1])

    def test_GeneratePythonFunction_Valid(self):
        """
        Event:
            - Call function 'generate_python_function' with valid definition.
        Evaluate:
            - Evaluate return type is tuple
            - Evaluate func is always None. (position 0)
            - Evaluate trace. (position 1)
        """
        with self.subTest(f"{self.id()}_0"):
            # Event
            result = generate_python_function("def Function1(t=0):\n\treturn 0", None)

            # Evaluate
            self.assertIsInstance(result, tuple)
            self.assertTrue(callable(result[0]))
            self.assertEqual(None, result[1])

        for f in (
            Function1,
            gray2dec,
            maximum,
            rpm_to_rad_per_second,
        ):
            with self.subTest(f"{self.id}_{f.__name__}"):
                source = inspect.getsource(f)
                # Event
                result = generate_python_function(source, None)

                # Evaluate
                self.assertIsInstance(result, tuple)
                self.assertTrue(callable(result[0]))
                self.assertEqual(None, result[1])
