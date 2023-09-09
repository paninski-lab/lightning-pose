import unittest
import pandas as pd
from your_module import call_active_all  
class TestActiveLearning(unittest.TestCase):

    def test_dataframe_consistency(self):
        active_cfg = 
        result1 = call_active_all(active_cfg)

        
        active_cfg['iteration_0']['method'] = 'other_method'
        result2 = call_active_all(active_cfg)

        
        self.assertTrue(result1['iteration_0']['csv_file_prev_run'] == result2['iteration_0']['csv_file_prev_run'])

if __name__ == '__main__':
    unittest.main()
