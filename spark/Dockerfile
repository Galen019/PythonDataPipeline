# Spark job submitter

FROM bitnami/spark:latest

USER root

# Create the missing directory and install Python
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

USER 1001

# Copy the normalization script
COPY normalization.py /opt/spark-apps/normalization.py

# Set the entrypoint to the Bitnami Spark entrypoint script
ENTRYPOINT ["/opt/bitnami/scripts/spark/entrypoint.sh"]

# Default command to run spark-submit
CMD ["spark-submit", "--master", "spark://spark-master:7077", "/opt/spark-apps/normalization.py"]