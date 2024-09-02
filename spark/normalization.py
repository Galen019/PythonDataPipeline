from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, max as spark_max


def normalize_dataframe(df):
    # Compute the minimum and maximum values of the "Value" column
    min_value = df.select(spark_min(col("Value"))).collect()[0][0]
    max_value = df.select(spark_max(col("Value"))).collect()[0][0]

    # Normalize the "Value" column to the range [0, 1]
    df_normalized = df.withColumn(
        "NormalizedValue", (col("Value") - min_value) / (max_value - min_value)
    )

    # Show the normalized DataFrame
    print("Normalized DataFrame:")
    df_normalized.show()

    return df_normalized


if __name__ == "__main__":
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("ExampleApp")  # type: ignore
        .master("spark://localhost:7077")
        .getOrCreate()
    )

    data = [(i,) for i in range(1001)]
    df = spark.createDataFrame(data, ["Value"])

    # Cache dataframe so subsequent operations are faster
    df.cache()

    # Show the original DataFrame
    print("Original DataFrame:")
    df.show()

    # Show the number of partitions
    print("Number of partitions:", df.rdd.getNumPartitions())

    normalize_dataframe(df)

    spark.stop()
