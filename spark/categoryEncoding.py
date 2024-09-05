from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.functions import col, mean


# Initialize Spark session
spark: SparkSession = (
    SparkSession.builder.appName("reduceByKey Example")  # type: ignore
    .master("spark://localhost:7077")
    .getOrCreate()
)


# Sample data with missing values in the 'category' column
data = [
    ("apple", 1, "fruit"),
    ("banana", 1, "fruit"),
    ("carrot", 1, "vegetable"),
    ("dog", 1, "animal"),
    ("cat", 1, "animal"),
    ("rose", 1, "flower"),
    ("tulip", 1, "flower"),
    ("daisy", 1, "flower"),
    ("unknown", None, None),  # Missing values
]

# Create a DataFrame
df = spark.createDataFrame(data, ["word", "count", "category"])

# Data Sanitization: Handling Missing Values in Categorical Variables
# Step 1: Impute missing category with the most frequent category
mode_category = (
    df.groupBy("category").count().orderBy("count", ascending=False).first()[0]  # type: ignore
)
df = df.fillna({"category": mode_category})

# Impute missing counts with mean of all counts
mean_count = df.select(mean(col("count"))).first()[0]  # type: ignore
df = df.fillna({"count": mean_count})

# Step 2: Index the categorical column 'category'
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
df_indexed = indexer.fit(df).transform(df)

# Step 3: One-Hot Encode the indexed column
encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
df_encoded = encoder.fit(df_indexed).transform(df_indexed)

# Show the results
df_encoded.select("word", "count", "category", "categoryIndex", "categoryVec").show(
    truncate=False
)

# Stop the Spark session
spark.stop()
