import unittest
from pyspark.sql import SparkSession
from spark.normalization import normalize_dataframe


class NormalizationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("TestNormalization")  # type: ignore
            .master("spark://localhost:7077")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark session after tests are completed
        cls.spark.stop()

    def test_normalize_dataframe(self):
        # Create a DataFrame with test data
        data = [(0,), (500,), (1000,)]
        df = self.spark.createDataFrame(data, ["Value"])

        # Apply the normalization function
        df_normalized = normalize_dataframe(df)
        result = df_normalized.collect()

        # Define the expected output
        expected_data = [(0, 0.0), (500, 0.5), (1000, 1.0)]
        expected_df = self.spark.createDataFrame(
            expected_data, ["Value", "NormalizedValue"]
        )
        expected_result = expected_df.collect()

        print("Expected: ")
        print(expected_result)
        print("Acutal: ")
        print(result)

        # Assert that the result matches the expected output
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
