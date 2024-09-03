from pyspark.sql import SparkSession
from pyspark import SparkContext, RDD


def reduce(rdd: RDD):
    # Transform the RDD to (key, (count, category)) pairs
    rdd_transformed = rdd.map(lambda x: (x[0], (x[1], x[2])))

    # Use reduceByKey to sum the counts for each word and keep the category
    result = rdd_transformed.reduceByKey(lambda x, y: (x[0] + y[0], x[1]))

    # Collect and print the results
    resultList = result.collect()
    print(resultList)

    return resultList


if __name__ == "__main__":
    # Initialize Spark session
    spark: SparkSession = (
        SparkSession.builder.appName("reduceByKey Example")  # type: ignore
        .master("spark://localhost:7077")
        .getOrCreate()
    )

    sc: SparkContext = spark.sparkContext

    # Sample data: list of (word, count, category)
    data = [
        ("apple", 1, "fruit"),
        ("banana", 1, "fruit"),
        ("apple", 1, "fruit"),
        ("apple", 1, "fruit"),
        ("banana", 1, "fruit"),
        ("cherry", 1, "fruit"),
    ]

    reduce(sc.parallelize(data))

    spark.stop()
