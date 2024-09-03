FROM bitnami/spark:latest

USER root

# Install numpy
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install numpy && \
    pip3 install pyspark && \
    rm -rf /var/lib/apt/lists/*

USER 1001

# Set the entrypoint to start the Spark worker
ENTRYPOINT ["/opt/bitnami/spark/bin/spark-class", "org.apache.spark.deploy.worker.Worker", "spark://spark-master:7077"]