import numpy as np
import unittest

import ta_abstract

class TaLibAbstractTestCase(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        ta_abstract.finalize()

    def test_function_groups(self):

        res = ta_abstract.function_groups()

        self.assertTrue(len(res) > 0)

    def test_functions_in_group(self):

        res = ta_abstract.functions_in_group('Math Operators')

        self.assertIsNotNone(res)
        
        print res

    def test_ta_function_class(self):

        fun = ta_abstract.TaFunction('MA')

        self.assertIsNotNone(fun)

        self.assertEquals('MA', fun.name)
        self.assertEquals('Overlap Studies', fun.group)
        self.assertEquals('Moving average', fun.hint)
        self.assertEquals(1, fun.nb_input)
        self.assertEquals(1, fun.nb_output)

        self.assertEquals(fun.nb_input, len(fun.input_description))

        parameter = fun.input_description[0]
        self.assertEquals('inReal', parameter.name)
        self.assertEquals('Real', parameter.type)
        self.assertEquals(0, parameter.flags)

        self.assertEquals(fun.nb_output, len(fun.output_description))

        parameter = fun.output_description[0]
        self.assertEquals('outReal', parameter.name)
        self.assertEquals('Real', parameter.type)
        self.assertEquals(ta_abstract.OUT_LINE, parameter.flags)


    def test_call_function(self):

        add_function = ta_abstract.TaFunction('ADD')

        input1 = np.sin(np.linspace(0, 2*np.pi))
        input2 = np.sin(np.linspace(0, 2*np.pi))

        start = 0
        end = 10

        results, begin, count = add_function(start, end, input1, input2)

        expected_result_count = add_function.nb_output

        self.assertEquals(expected_result_count, len(results))

        np.testing.assert_almost_equal(  (input1 + input2)[begin:begin+count], results[0])

    def test_optional_parameters(self):

        ma_function = ta_abstract.TaFunction('MA')

        optional_parameters = ma_function.optional_input_description

        self.assertTrue(len(optional_parameters) > 0)

        print optional_parameters

    def test_optional_parameter_call(self):

        ma_function = ta_abstract.TaFunction('MA')

        input1 = np.arange(100, dtype=float) + np.random.randint(0,10, size=100)

        results, begin, count = ma_function(0, 100, input1, optInTimePeriod=12)

        print input1, results[0], begin, count


if __name__ == '__main__':
    unittest.main()

