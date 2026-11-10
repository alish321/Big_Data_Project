# Scalable GPS Route Clustering (COMP4124 – Group 4)

A PySpark pipeline that turns raw taxi GPS logs into trips, encodes them as spatial-cell sequences, finds similar routes at scale, clusters them, and scores unusual trips.

**Notebook:** `Group4_COMP4124_Code.ipynb`
**Dataset:** Beijing taxi GPS data (T-Drive). Public copy: <https://www.kaggle.com/datasets/arashnic/tdriver> (A. Niculescu, Kaggle).

---

## Pipeline overview

```
Raw GPS → cleaning → trip construction → segmentation → spatial partitioning (Grid / Quadtree / KDB-tree)
→ cell sequences → OD grouping → candidate pairs → route similarity (Jaccard / LCSS / Edit)
→ frequent routes → similarity graph → label propagation → anomaly scoring → evaluation → scalability
```

| # | Stage | What it does |
|---|-------|--------------|
| 01 | Bronze ingestion | Reads raw CSV (no header), normalises to `taxi_id, event_time, lon, lat` |
| 02 | GPS cleaning | Drops duplicates, nulls, zero/invalid coordinates, points outside the Beijing bounding box; removes impossible speeds and non-increasing timestamps; prints a quality report |
| 03 | Trip construction | Starts a new trip on a time gap, large jump, or impossible speed; keeps trips with ≥ 2 points |
| 04 | Segment generation | Per-segment distance, duration, speed, heading, turn angle, acceleration |
| 05 | Spatial partitioning evaluation | Compares uniform grid, quadtree and KDB-tree on partition count, size spread and skew ratio |
| 06 | Cell sequences | Maps points to ~0.005° grid cells, collapses repeated cells, derives origin/destination (OD) pair and trip distance/speed stats |
| 07 | Candidate pair generation | Blocks by OD pair and shared cells, then prunes by distance and duration ratio to avoid all-pairs comparison |
| 08 | Route similarity + mining | Jaccard, LCSS and edit-distance similarity combined into one score; n-gram (3–5) frequent route patterns |
| 09 | Graph and clustering | Cell-to-cell route graph, trip similarity graph, label-propagation clusters, cluster summaries and sizes |
| 10 | Anomaly scoring | Interpretable route-based anomaly score with human-readable reasons; top 5% flagged |
| 11 | Evaluation | Row counts per stage, intra- vs inter-cluster similarity, popular OD pairs and routes, per-stage runtimes |
| – | Scalability benchmark | Times candidate generation, similarity, graph build and clustering on 10k / 50k / 100k trips |

## Key methods

- **Distance and heading:** Haversine distance and bearing implemented with Spark column expressions.
- **Spatial partitioning:** uniform 10×10 grid, quadtree (split until ≤ 100k points or depth 5), KDB-tree (median split on the longer axis, ≤ 300k points or depth 4).
- **Route similarity:** `0.40 × Jaccard + 0.40 × LCSS + 0.20 × edit similarity`. LCSS and edit distance run as Python UDFs.
- **Clustering:** label propagation (minimum neighbour label) over trip pairs with similarity ≥ 0.45.
- **Anomaly score:** `0.30 × route rarity + 0.25 × graph-edge rarity + 0.20 × duration deviation + 0.15 × distance deviation + 0.10 × speed z-score`.

## Default configuration (`PipelineConfig`)

| Parameter | Value |
|-----------|-------|
| Bounding box | lon 115.0–117.5, lat 39.0–41.0 |
| `TRIP_GAP_MINUTES` | 10 |
| `MAX_POINT_JUMP_KM` / `MAX_SPEED_KMH` | 5.0 / 160.0 |
| `CELL_SIZE_DEG` | 0.005 |
| `MIN_SHARED_CELLS` | 2 |
| `MAX_DISTANCE_RATIO_DIFF` / `MAX_DURATION_RATIO_DIFF` | 0.40 / 0.60 |
| `MIN_JACCARD_SIMILARITY` / `MIN_SEQUENCE_SIMILARITY` | 0.20 / 0.30 |
| `ROUTE_EDGE_SIMILARITY_THRESHOLD` | 0.45 |
| `RANDOM_SEED` | 42 |

Runtime flags near the top of the notebook:

| Flag | Default | Meaning |
|------|---------|---------|
| `DEMO_MODE` / `DEMO_LIMIT` | `False` / 15,000 | Limits the data used in the partitioning evaluation |
| `RUN_PARTITIONING_EVALUATION` | `True` | Run the Grid / Quadtree / KDB comparison |
| `RUN_SCALABILITY_BENCHMARK` | `True` | Run the 10k / 50k / 100k benchmark |
| `WRITE_OUTPUTS` | `False` | Write intermediate tables as Parquet |
| `OUTPUT_ROOT` | `/tmp/comp4124_taxi_outputs` | Override with the `TAXI_OUTPUT_ROOT` environment variable |

## Requirements

- Python 3 and **Apache Spark / PySpark** (the notebook uses `%md` cells and `display()`, so it was written for a Databricks-style environment; it falls back to `df.show()` elsewhere)
- Only PySpark and the Python standard library are imported
- Enough memory for large self-joins. Candidate-pair generation and the benchmark are the heaviest stages

## How to run

1. Get the T-Drive data and place it where Spark can read it.
2. Set `RAW_INPUT_PATH` in `PipelineConfig`. It currently points to an Azure storage location (`abfss://2026-group4@…/taxi_log_2008_by_id/`) that only the original team can access, so **you must change it**.
3. The input is headerless CSV: `taxi_id, timestamp, lon, lat`.
4. Optionally set `DEMO_MODE = True` or turn off the benchmark for a faster first run.
5. Run all cells top to bottom. The first half defines functions; the second half (**Pipeline execution**) runs the stages in order.
6. To keep results, set `WRITE_OUTPUTS = True` and read the Parquet tables from `OUTPUT_ROOT`.

## Outputs (when `WRITE_OUTPUTS = True`)

`bronze_raw`, `gps_clean`, `gps_validated`, `gps_quality_report`, `trips`, `trip_summary`, `segments`, `partition_metrics`, `trip_cell_sequences`, `od_trip_groups`, `candidate_trip_pairs`, `candidate_reduction`, `route_similarity`, `frequent_routes`, `route_graph_edges`, `trip_similarity_graph`, `route_clusters`, `route_cluster_summary`, `route_anomaly_scores`, `anomalous_trips`, `basic_count_metrics`, `cluster_quality`, `od_patterns`, `popular_routes`, `runtime_metrics`, `scalability_benchmark`, `scalability_benchmark_summary`.

## Known limitations

- The Dense and LSTM autoencoder anomaly scores are **placeholders set to 0.0**. The final anomaly score currently equals the route-based score only.
- Label propagation runs for 3 iterations in the main pipeline and 8 in the benchmark, so clusters are approximate.
- The cell-sequence section appears twice in the notebook (a near-identical redefinition). The later definition is the one the pipeline uses.
- Exact (cell-based) OD pairing means trips with slightly different endpoints land in different groups. Adjust `CELL_SIZE_DEG` to trade precision against grouping.
- Row counts and runtimes depend on the cluster and data size; none are hard-coded in this README.

## Project structure

```
Group4_COMP4124_Code.ipynb   # entire pipeline (functions + execution)
```
