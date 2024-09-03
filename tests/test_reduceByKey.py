import unittest
from pyspark.sql import SparkSession
from spark.reduceByKey import reduce


class ReduceByKeyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("TestReduceByKey")  # type: ignore
            .master("spark://localhost:7077")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark session after tests are completed
        cls.spark.stop()

    def test_reduceByKey(self):
        # Sample data: list of (word, count, category)
        data = [
            ("apple", 1, "fruit"),
            ("banana", 1, "fruit"),
            ("apple", 1, "fruit"),
            ("apple", 1, "fruit"),
            ("banana", 1, "fruit"),
            ("cherry", 1, "fruit"),
        ]

        sc = self.spark.sparkContext

        actual = reduce(sc.parallelize(data))

        expected = [
            ("apple", (3, "fruit")),
            ("banana", (2, "fruit")),
            ("cherry", (1, "fruit")),
        ]

        self.assertEqual(actual, expected)
