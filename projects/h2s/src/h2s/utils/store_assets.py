"""Simplified asset storage utilities for H2S project.

This is a lightweight version of the resilient_workflows_public store_assets
module, containing only the functions needed for H2S predictions without
heavy dependencies like pydantic_schemaorg, geopandas, foursquare, etc.
"""

import json
import os
from datetime import datetime
from typing import List

import pandas as pd
import pytz
from dagster import get_dagster_logger

from h2s.resources.minio import S3Resource


def getTodayAsIso():
    """Get current timestamp as ISO string in Pacific timezone."""
    return datetime.now(pytz.timezone("America/Los_Angeles")).isoformat()


def fix_col_types(df, date_format=None):
    """Fix datetime columns to string format for JSON serialization."""
    columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    for col in columns:
        get_dagster_logger().debug(f"fixing col {col} to {date_format}")
        df[col] = df[col].apply(
            lambda x: x.strftime(date_format)
            if pd.notnull(x) and date_format
            else x.isoformat()
            if pd.notnull(x)
            else None
        )
    return df


def addLastUpdatedRecords(json_str, date_str) -> str:
    """Add lastUpdated field to JSON records."""
    json_obj = json.loads(json_str)
    new_json = {"lastUpdated": date_str, "data": json_obj}
    return json.dumps(new_json, indent=2)


def get_latest_basepath() -> str:
    """Get the LATEST_BASEPATH environment variable with fallback."""
    latest = os.environ.get("LATEST_BASEPATH", "latest")
    if latest.endswith("/"):
        latest = latest[:-1]
    return latest


class SimpleMetadata:
    """Simplified metadata container without pydantic_schemaorg dependencies."""

    def __init__(self, name=None, description=None, variableMeasured=None):
        self.name = name
        self.alternateName = None
        self.description = description
        self.variableMeasured = variableMeasured or []
        self.distribution = []

    def copy(self):
        """Create a copy of this metadata object."""
        new_meta = SimpleMetadata(
            name=self.name,
            description=self.description,
            variableMeasured=self.variableMeasured.copy() if self.variableMeasured else []
        )
        new_meta.alternateName = self.alternateName
        new_meta.distribution = self.distribution.copy() if self.distribution else []
        return new_meta

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "alternateName": self.alternateName,
            "description": self.description,
            "variableMeasured": self.variableMeasured,
            "distribution": [
                {"encodingFormat": d["format"], "contentUrl": d["url"]}
                for d in self.distribution
            ],
        }

    def json(self, indent=2):
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def objectMetadata(name=None, description=None, variableMeasured=None) -> SimpleMetadata:
    """Create a simple metadata object for datasets.

    Args:
        name: Dataset name
        description: Dataset description
        variableMeasured: List of variable names

    Returns:
        SimpleMetadata object
    """
    return SimpleMetadata(name=name, description=description, variableMeasured=variableMeasured)


def metadata_to_s3(metadata: SimpleMetadata, path_w_basename, s3_resource: S3Resource):
    """Write metadata to S3 as JSON file."""
    s3_resource.putFile_text(
        data=metadata.json(indent=2), path=f"{path_w_basename}.metadata.json"
    )


def dataframe_to_s3(
    dataframe: pd.DataFrame,
    path_w_basename: str,
    s3_resource: S3Resource,
    formats=["csv", "json"],
    metadata=None,
):
    """Store DataFrame to S3 in multiple formats.

    Args:
        dataframe: DataFrame to store
        path_w_basename: S3 path without file extension
        s3_resource: S3Resource instance
        formats: List of formats to export ('csv', 'json', 'parquet')
        metadata: Optional SimpleMetadata object
    """
    date = getTodayAsIso()
    distributions = []

    for format in formats:
        if format == "json":
            df = fix_col_types(dataframe.copy())
            try:
                gdf_json = df.to_json(orient="records")
                new_json = addLastUpdatedRecords(gdf_json, date)

                path = f"{path_w_basename}.json"
                object_name = s3_resource.putFile_text(data=new_json, path=path)
                distributions.append({"format": "json", "url": object_name})
            except Exception as e:
                get_dagster_logger().info(f"dataframe_to_s3, failed to write json {e}")

        elif format == "csv":
            csv_data = dataframe.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
            path = f"{path_w_basename}.csv"
            object_name = s3_resource.putFile_text(data=csv_data, path=path)
            distributions.append({"format": "csv", "url": object_name})

        elif format == "parquet":
            path = f"{path_w_basename}.parquet"
            parquet_object = dataframe.to_parquet()
            object_name = s3_resource.putFile(
                data=parquet_object, path=path, content_type="application/vnd.apache.parquet"
            )
            distributions.append({"format": "parquet", "url": object_name})

    if metadata is not None:
        metadata.distribution = distributions
        metadata_to_s3(metadata, path_w_basename, s3_resource)


def store_dataframe_to_s3(
    df: pd.DataFrame,
    path: str,
    dataset_identifier: str,
    s3_resource: S3Resource,
    metadata=None,
    latestdatasetpath=None,
    enable_latest_path: bool = False,
    formats=["csv", "json"],
):
    """Store DataFrame to S3 with optional latest path copy.

    Args:
        df: DataFrame to store
        path: Base S3 path (without filename)
        dataset_identifier: Dataset filename (without extension)
        s3_resource: S3Resource instance
        metadata: Optional SimpleMetadata object
        latestdatasetpath: Path for latest copy (relative to latest/)
        enable_latest_path: Whether to also store in latest/ path
        formats: List of formats to export
    """
    logger = get_dagster_logger()

    # Clean paths
    if latestdatasetpath and latestdatasetpath.endswith("/"):
        latestdatasetpath = latestdatasetpath[:-1]
    if path and path.endswith("/"):
        path = path[:-1]

    path_w_basename = f"{path}/{dataset_identifier}"

    if enable_latest_path:
        latestdatasetpath_basename = (
            f"{get_latest_basepath()}/{latestdatasetpath}/{dataset_identifier}"
        )
        latest_metadata = metadata.copy() if metadata else None
        if latest_metadata:
            latest_metadata.name = f"latest {metadata.name}"
            if latest_metadata.alternateName:
                latest_metadata.alternateName = f"latest {metadata.alternateName}"
            latest_metadata.description = f"latest {metadata.description}"

    # Store in original path
    dataframe_to_s3(df, path_w_basename, s3_resource, metadata=metadata, formats=formats)
    logger.info(
        f"Stored DataFrame for {dataset_identifier} to S3: s3://{s3_resource.S3_BUCKET}/{path_w_basename}"
    )

    # Store in latest path
    if enable_latest_path:
        dataframe_to_s3(
            df, latestdatasetpath_basename, s3_resource, metadata=latest_metadata, formats=formats
        )
        logger.info(
            f"Stored DataFrame for {dataset_identifier} to latest path: s3://{s3_resource.S3_BUCKET}/{latestdatasetpath_basename}"
        )
