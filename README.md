Real-Time Multi-Domain Big Data Analytics for Smart Cities
Overview
This repository contains my research paper that proposes a scalable real-time big data analytics framework designed for smart city environments.

Modern smart cities generate massive data streams from multiple domains such as transportation, environment, energy, and public safety.
This research presents an integrated architecture capable of collecting, processing, storing, and analyzing high-volume IoT data in real time to enable intelligent decision-making.

Problem Statement
Most existing smart city solutions operate in data silos, making cross-domain analytics difficult and limiting the potential of data-driven governance.

This research addresses:

Real-time processing of multi-domain IoT data
Scalable storage and distributed computing
Predictive analytics for smart city insights
Unified architecture for cross-domain integration
Proposed Solution
The system introduces a real-time analytics pipeline that:

Collects IoT data streams from multiple domains
Processes data using distributed streaming frameworks
Stores structured and unstructured data efficiently
Applies machine learning models for prediction and insights
System Architecture
Data Ingestion
Apache Kafka handles high-throughput real-time data streams from IoT devices.
Stream Processing
Apache Spark Structured Streaming performs:
Data filtering
Aggregation
Pattern detection
Data Storage
Hadoop HDFS → Scalable distributed storage
MongoDB → Fast querying and flexible document storage
Machine Learning Layer
Scikit-learn models used for predictive analytics and intelligent insights.
Technologies Used
Apache Kafka
Apache Spark Structured Streaming
Hadoop HDFS
MongoDB
Scikit-learn
Big Data Analytics
Machine Learning
   Key Contributions
✔ Multi-domain data integration architecture
✔ Real-time analytics pipeline for smart cities
✔ Scalable big data storage and processing
✔ Machine learning-based predictive insights

Author
Mannat
