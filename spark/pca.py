from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import format_number, col
import numpy as np


# Initialize Spark session
spark = (
    SparkSession.builder.appName("PCA Example")  # type: ignore
    .master("spark://localhost:7077")
    .getOrCreate()
)

# Generate synthetic data with 300 dimensions
np.random.seed(42)
data = [
    (i, Vectors.dense(np.random.rand(300))) for i in range(100)
]  # 100 samples with 300 features each

# Create a DataFrame
df: DataFrame = spark.createDataFrame(data, ["id", "features"])

# Introduce some missing values for demonstration
data[5] = (5, Vectors.dense([np.nan] * 300))

# Data Sanitization
# 1. Remove rows with missing values
df_cleaned = df.filter(~col("features").isNull())

# Perform PCA to reduce dimensions to 2
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(df)
result: DataFrame = model.transform(df).select("id", "pca_features")

# Show the results
result.show(truncate=False)

# Stop the Spark session
spark.stop()
