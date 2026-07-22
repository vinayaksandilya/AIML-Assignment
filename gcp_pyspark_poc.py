import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg as spark_avg, sum as spark_sum, count as spark_count,
    to_date, substring, year, month, when, round as spark_round
)

def parse_args():
    parser = argparse.ArgumentParser(description="MES PySpark Deployment on GCP Dataproc")
    # Hardcoded default bucket name: vinayakbucket
    parser.add_argument("--data_bucket", default="gs://vinayakbucket", help="GCS Bucket Name (default: gs://vinayakbucket)")
    return parser.parse_args()

def main():
    args = parse_args()
    bucket_path = args.data_bucket.rstrip('/')

    # 1. Initialize Spark Session with GCS & Legacy Time Parser configurations
    spark = SparkSession.builder \
        .appName("MES_GCP_Dataproc_SmartMeter_Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()

    print("\n==================================================================")
    print(f"=== STARTING DATAPROC SPARK PIPELINE FOR BUCKET: {bucket_path} ===")
    print("==================================================================\n")

    # 2. Ingest daily_dataset.csv from GCS root data folder
    daily_dataset_path = f"{bucket_path}/data/daily_dataset.csv"
    print(f"[STAGE 1] Ingesting daily smart meter dataset from: {daily_dataset_path}")
    
    raw_df = spark.read.option("header", "true").csv(daily_dataset_path)
    raw_count = raw_df.count()
    print(f"  -> Raw Ingested Energy Records: {raw_count:,}")

    # 3. Veracity Cleaning & Type Casting
    clean_df = raw_df \
        .withColumn("day_str", substring(col("day"), 1, 10)) \
        .withColumn("day_ts", to_date(col("day_str"), "yyyy-MM-dd")) \
        .withColumn("energy_kwh", col("energy_sum").cast("double")) \
        .where(col("energy_kwh").isNotNull() & (col("energy_kwh") >= 0))

    clean_count = clean_df.count()
    dropped_count = raw_count - clean_count
    print(f"  -> Cleaned Veracity Energy Records: {clean_count:,} (Dropped {dropped_count} invalid rows)")

    # 4. Ingest Household Demographic Metadata (informations_households.csv)
    household_path = f"{bucket_path}/data/informations_households.csv"
    print(f"\n[STAGE 2] Ingesting household metadata from: {household_path}")
    hh_df = spark.read.option("header", "true").csv(household_path) \
        .select("LCLid", "stdorToU", "Acorn_grouped")

    # Join energy data with household demographics
    clean_df = clean_df.join(hh_df, on="LCLid", how="left")

    # --- DEMOGRAPHIC & TARIFF ANALYTICS ---
    print(f"\n[ANALYSIS 1] Computing Tariff & Demographic Demand Profiles...")
    tariff_summary = clean_df.groupBy("stdorToU") \
        .agg(
            spark_round(spark_avg("energy_kwh"), 3).alias("mean_daily_kwh"),
            spark_count("energy_kwh").alias("record_count")
        ).orderBy("stdorToU").collect()

    acorn_summary = clean_df.groupBy("Acorn_grouped") \
        .agg(
            spark_round(spark_avg("energy_kwh"), 3).alias("mean_daily_kwh"),
            spark_count("energy_kwh").alias("record_count")
        ).orderBy("Acorn_grouped").collect()

    # 5. Aggregate Daily Household Consumption
    print(f"\n[STAGE 3] Performing daily spatial aggregations across households...")
    daily_avg_df = clean_df \
        .groupBy("day_ts") \
        .agg(spark_avg("energy_kwh").alias("mean_daily_kwh")) \
        .orderBy("day_ts")

    # 6. Ingest Weather Data (weather_daily_darksky.csv)
    weather_path = f"{bucket_path}/data/weather_daily_darksky.csv"
    print(f"\n[STAGE 4] Ingesting weather records from: {weather_path}")
    
    weather_df = spark.read.option("header", "true").csv(weather_path) \
        .withColumn("time_str", substring(col("time"), 1, 10)) \
        .withColumn("day_ts", to_date(col("time_str"), "yyyy-MM-dd")) \
        .withColumn("temp_max", col("temperatureMax").cast("double")) \
        .withColumn("temp_min", col("temperatureMin").cast("double")) \
        .withColumn("avg_temp", (col("temp_max") + col("temp_min")) / 2.0) \
        .withColumn("humidity", col("humidity").cast("double")) \
        .withColumn("windSpeed", col("windSpeed").cast("double"))

    # 7. Ingest UK Bank Holidays (uk_bank_holidays.csv)
    holiday_path = f"{bucket_path}/data/uk_bank_holidays.csv"
    print(f"\n[STAGE 5] Ingesting UK bank holidays from: {holiday_path}")
    
    holiday_df = spark.read.option("header", "true").csv(holiday_path) \
        .withColumn("h_str", substring(col("Bank holidays"), 1, 10)) \
        .withColumn("day_ts", to_date(col("h_str"), "yyyy-MM-dd")) \
        .withColumn("is_holiday", when(col("Type").isNotNull(), 1).otherwise(0)) \
        .select("day_ts", "is_holiday", col("Type").alias("holiday_name"))

    # 8. Multi-Source Integration Join
    print(f"\n[STAGE 6] Executing multi-source integration join...")
    merged_df = daily_avg_df \
        .join(weather_df.select("day_ts", "avg_temp", "humidity", "windSpeed"), on="day_ts", how="inner") \
        .join(holiday_df, on="day_ts", how="left")

    # Default missing holiday flags to 0 and 'Standard Day'
    merged_df = merged_df.fillna({"is_holiday": 0, "holiday_name": "Standard Day"})

    # Derive explicit year and month columns for partitioning and seasonal analysis
    merged_df = merged_df \
        .withColumn("year", year("day_ts")) \
        .withColumn("month", month("day_ts")) \
        .withColumn("mean_daily_kwh", spark_round(col("mean_daily_kwh"), 4)) \
        .withColumn("avg_temp", spark_round(col("avg_temp"), 2))

    merged_count = merged_df.count()
    print(f"  -> Merged Daily Energy-Weather-Holiday Records: {merged_count:,}")

    # --- ADVANCED STATISTICAL & SEASONAL ANALYTICS ---
    print(f"\n[ANALYSIS 2] Computing Pearson Correlation & Seasonal Load Dynamics...")
    
    # Pearson Correlation (r) between avg_temp and mean_daily_kwh
    r_stat = merged_df.stat.corr("mean_daily_kwh", "avg_temp")

    # Seasonal Averages (Winter: Dec, Jan, Feb; Summer: Jun, Jul, Aug)
    winter_df = merged_df.filter(col("month").isin([12, 1, 2]))
    summer_df = merged_df.filter(col("month").isin([6, 7, 8]))
    
    winter_mean = winter_df.agg(spark_avg("mean_daily_kwh")).collect()[0][0] or 0.0
    summer_mean = summer_df.agg(spark_avg("mean_daily_kwh")).collect()[0][0] or 0.0
    ws_ratio = winter_mean / summer_mean if summer_mean > 0 else 0.0

    # Bank Holiday vs Standard Day Averages
    holiday_mean = merged_df.filter(col("is_holiday") == 1).agg(spark_avg("mean_daily_kwh")).collect()[0][0] or 0.0
    std_day_mean = merged_df.filter(col("is_holiday") == 0).agg(spark_avg("mean_daily_kwh")).collect()[0][0] or 0.0

    # Top 5 Peak Consumption Days
    peak_days = merged_df.orderBy(col("mean_daily_kwh").desc()).limit(5).collect()

    # --- BUILD FORMATTED TEXT REPORT ---
    report_lines = []
    report_lines.append("==================================================================================")
    report_lines.append("        METROENERGY SOLUTIONS (MES) SPARK BIG DATA ANALYTICAL REPORT             ")
    report_lines.append("==================================================================================")
    report_lines.append(f"Target GCS Storage Bucket : {bucket_path}")
    report_lines.append(f"Total Raw Ingested Rows   : {raw_count:,}")
    report_lines.append(f"Cleaned Veracity Rows     : {clean_count:,} (Dropped {dropped_count} invalid rows)")
    report_lines.append(f"Merged Timeline Horizon   : {merged_count:,} Daily Records")
    report_lines.append("----------------------------------------------------------------------------------")
    report_lines.append("\n1. DEMOGRAPHIC & TARIFF CONSUMPTION PROFILES:")
    report_lines.append("   Tariff Type (stdorToU):")
    for row in tariff_summary:
        report_lines.append(f"     * {row['stdorToU']:<8} : Mean Consumption = {row['mean_daily_kwh']:.3f} kWh/day | Records = {row['record_count']:,}")
    
    report_lines.append("\n   Acorn Social Group (Acorn_grouped):")
    for row in acorn_summary:
        group_name = row['Acorn_grouped'] if row['Acorn_grouped'] else 'Unknown'
        report_lines.append(f"     * {group_name:<12} : Mean Consumption = {row['mean_daily_kwh']:.3f} kWh/day | Records = {row['record_count']:,}")

    report_lines.append("\n2. SEASONAL & ENVIRONMENTAL LOAD ANALYSIS:")
    report_lines.append(f"   * Pearson Correlation (r) [Temperature vs Energy Demand] : {r_stat:.4f}")
    report_lines.append(f"   * Mean Winter Household Consumption (Dec, Jan, Feb)     : {winter_mean:.3f} kWh/day")
    report_lines.append(f"   * Mean Summer Household Consumption (Jun, Jul, Aug)     : {summer_mean:.3f} kWh/day")
    report_lines.append(f"   * Winter-to-Summer Demand Ratio                         : {ws_ratio:.2f}x (+{(ws_ratio-1)*100:.1f}% Winter Demand Surge)")
    report_lines.append(f"   * Mean Bank Holiday Consumption                          : {holiday_mean:.3f} kWh/day")
    report_lines.append(f"   * Mean Standard Working Day Consumption                 : {std_day_mean:.3f} kWh/day")

    report_lines.append("\n3. TOP 5 PEAK CONSUMPTION DAYS IDENTIFIED:")
    for idx, row in enumerate(peak_days, 1):
        report_lines.append(f"   #{idx}. Date: {row['day_ts']} | Demand: {row['mean_daily_kwh']:.2f} kWh/day | Temp: {row['avg_temp']:.2f}°C | {row['holiday_name']}")

    report_lines.append("==================================================================================")

    report_text = "\n".join(report_lines)
    print("\n" + report_text + "\n")

    # 9. Write Partitioned Output to GCS in Parquet Format
    parquet_output_path = f"{bucket_path}/output/daily_energy_weather_parquet"
    print(f"[STAGE 7] Writing partitioned Parquet table to: {parquet_output_path}")
    
    merged_df.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(parquet_output_path)

    # 10. Save Text Report File to GCS
    txt_report_path = f"{bucket_path}/output/analytical_summary_report_txt"
    print(f"[STAGE 8] Writing human-readable text summary report to: {txt_report_path}")
    
    # Save text report into single partition text file in GCS
    spark.sparkContext.parallelize([report_text]).coalesce(1).saveAsTextFile(txt_report_path)

    print(f"\n==================================================================")
    print(f"=== PIPELINE & ANALYTICS COMPLETED SUCCESSFULLY IN DATAPROC ===")
    print(f"=== Parquet Output Table: {parquet_output_path}")
    print(f"=== TXT Report Output   : {txt_report_path}")
    print(f"==================================================================\n")
    
    spark.stop()

if __name__ == "__main__":
    main()
