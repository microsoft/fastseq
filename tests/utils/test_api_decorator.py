# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from absl.testing import absltest, parameterized
from fastseq.utils.api_decorator import get_class, override_method, add_method, export_api, replace
from fastseq.utils.test_utils import TestCaseBase


class A:
    def name(self):
        return 'A'


def func_a():
    return 'a'


class APIDecoratorTest(TestCaseBase):
    def disable_test_get_class(self):
        self.assertEqual(get_class(A.name), A)

        class B:
            def name(self):
                return 'B'

        self.assertEqual(get_class(B.name), APIDecoratorTest)

    def test_override_method(self):
        @override_method(A.name)
        def name_c(self):
            return 'B'

        a = A()
        self.assertEqual(a.name(), 'B')

    def test_add_method(self):
        class A:
            def name(self):
                return 'A'

        @add_method(A)
        def area(self):
            return 1

        a = A()
        self.assertEqual(a.area(), 1)

    def disable_test_export_api(self):
        # export a new api.
        @export_api("test.export.api", "B")
        class B:
            def name(self):
                return 'B'

        from test.export.api import B
        b = B()
        self.assertEqual(b.name(), 'B')

        # the export api already exists.
        @export_api("test.export.api", "B")
        class C:
            def name(self):
                return 'C'

        from test.export.api import B
        b = B()
        self.assertEqual(b.name(), 'C')

    def test_replace(self):
        # replace a class.
        @replace(A)
        class B:
            def name(self):
                return ('test_replace_B')

        a = A()
        self.assertEqual(a.name(), 'test_replace_B')

        # replace a function.
        @replace(func_a)
        def func_b():
            return 'b'

        self.assertEqual(func_a(), 'b')


if __name__ == "__main__":
    absltest.main()
