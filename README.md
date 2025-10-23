# Pyspark_Clustering
This repository contains the solution for Assignment 2 in the Big Data course.  
The project implements the K-Means clustering algorithm using Apache Spark (PySpark) with the MapReduce paradigm.  
The solution was developed and tested on Databricks Community Edition.

## Project Structure
kmeans_clustering_pyspark.ipynb – PySpark implementation of the K-Means algorithm  
kmeans_report.pdf – Algorithm documentation and evaluation  
iris.csv – Dataset 1  
glass.csv – Dataset 2  
parkinsons.csv – Dataset 3  

---

## Part 1: Algorithm Overview
The K-Means algorithm is an unsupervised machine learning technique used to group data points into K clusters based on feature similarity.

### Algorithm Steps
1. Randomly select K initial centroids from the dataset.
2. For each data point:
   - Compute the Euclidean distance to each centroid.
   - Assign the point to the nearest centroid.
3. For each cluster:
   - Recalculate the centroid as the average of all points assigned to it.
4. Repeat steps 2–3 until:
   - The centroid changes are below the convergence threshold (CT), or
   - The maximum number of iterations (I) is reached.

Each experiment is repeated 10 times (Exp = 10) for every K value to minimize randomness in initialization.

---

## Part 2: Implementation Details

### Tools and Libraries
- Apache Spark (PySpark)
- Python 3
- scikit-learn (MinMaxScaler)
- pandas, NumPy

### Data Processing Flow
1. Read the input dataset (CSV) into a Spark DataFrame.
2. Normalize the data using MinMaxScaler.
3. Convert the DataFrame into an RDD for distributed processing.
4. Apply the MapReduce operations:
   - **map**: Assign each data point to the nearest centroid.  
     Example:
     ```python
     data_rdd.map(assign_cluster)
     ```
   - **filter**: Select only points belonging to a specific cluster.  
     ```python
     classification.filter(lambda obs: check(obs, j))
     ```
   - **reduce**: Compute the sum of all points in the cluster to derive the new centroid.  
     ```python
     cluster_j.reduce(sum_points)
     ```
5. Recalculate centroids and check for convergence.

### Data Structures
- **RDDs** – Core distributed structure for map, filter, and reduce operations.
- **Lists/Dictionaries** – Used for centroid management and intermediate storage.
- **DataFrames** – Used for initial data loading and normalization.

---

## Part 3: Parameters
| Parameter | Default | Description |
|------------|----------|-------------|
| K | 2–6 | Number of clusters |
| CT | 0.0001 | Convergence threshold |
| I | 30 | Maximum number of iterations per experiment |
| Exp | 10 | Number of repeated experiments |

Each experiment stops when convergence is achieved or the iteration limit is reached.

---

## Part 4: Evaluation
The algorithm was evaluated on the following datasets:
- iris.csv  
- glass.csv  
- parkinsons.csv  

### Metrics
1. **Calinski–Harabasz (CH) Score** – Ratio between inter-cluster and intra-cluster dispersion.  
2. **Adjusted Rand Index (ARI)** – Measures similarity between predicted and true cluster labels.

### Output Format
Average and Std of CH: (XXX ; YYY)
Average and Std of ARI: (XXX ; YYY)

---

## Part 5: How to Run
1. Open the notebook `kmeans_clustering_pyspark.ipynb` in Databricks or Google Colab.  
2. Upload the dataset files:
   - iris.csv
   - glass.csv
   - parkinsons.csv  
3. Run all notebook cells sequentially.  
4. The algorithm will print the average and standard deviation of CH and ARI for each dataset and K value.

---

## Part 6: Results Summary
- **Iris dataset**: High CH and ARI scores – clear and well-separated clusters.  
- **Glass dataset**: Moderate results due to overlapping features.  
- **Parkinsons dataset**: Lower stability due to dataset complexity.  

---

## Part 7: Conclusion
The project demonstrates a distributed and scalable implementation of the K-Means algorithm using PySpark.  
By combining Spark RDDs with the MapReduce paradigm, the algorithm efficiently processes large datasets and provides accurate clustering results.
