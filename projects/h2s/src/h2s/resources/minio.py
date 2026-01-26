"""S3/MinIO resource for Dagster integration."""

import io
from typing import Iterator

import minio.datatypes
from dagster import ConfigurableResource, get_dagster_logger
from minio import Minio
from pydantic import Field


def PythonMinioAddress(url, port=None):
    """Format MinIO address with optional port."""
    if url.endswith(".amazonaws.com"):
        PYTHON_MINIO_URL = "s3.amazonaws.com"
    else:
        PYTHON_MINIO_URL = url
    if port is not None:
        PYTHON_MINIO_URL = f"{PYTHON_MINIO_URL}:{port}"
    return PYTHON_MINIO_URL


class ResourceWithS3Configuration(ConfigurableResource):
    """Base resource with S3 configuration."""

    S3_BUCKET: str = Field(description="S3_BUCKET.")
    S3_ADDRESS: str = Field(description="S3_HOST NAME.")
    S3_PORT: str = Field(description="S3_PORT.")
    S3_USE_SSL: bool = Field(default=True)
    S3_ACCESS_KEY: str = Field(description="S3_ACCESS_KEY")
    S3_SECRET_KEY: str = Field(description="S3_SECRET_KEY")


class S3Resource(ResourceWithS3Configuration):
    """S3/MinIO resource for object storage operations."""

    def MinioOptions(self):
        """Get MinIO client options."""
        return {
            "secure": self.S3_USE_SSL,
            "access_key": self.S3_ACCESS_KEY,
            "secret_key": self.S3_SECRET_KEY,
        }

    def getClient(self):
        """Get MinIO client instance."""
        return Minio(
            PythonMinioAddress(self.S3_ADDRESS, self.S3_PORT),
            self.S3_ACCESS_KEY,
            self.S3_SECRET_KEY,
        )

    def baseUrl(self):
        """Get base URL for S3 endpoint."""
        url = PythonMinioAddress(self.S3_ADDRESS, self.S3_PORT)
        if self.S3_USE_SSL:
            return f"https://{url}"
        else:
            return f"http://{url}"

    def listPath(
        self, path="orgs", recusrsive=True, bucket=None
    ) -> Iterator[minio.datatypes.Object]:
        """List objects at a given path."""
        if bucket is None:
            bucket = self.S3_BUCKET
        result = self.getClient().list_objects(bucket, path)
        return result

    def publicUrl(self, path="test", bucket=None):
        """Get public URL for an object."""
        return f"{self.baseUrl()}/{bucket}/{path}"

    def getFile(self, path="test", bucket=None):
        """Get file data from S3."""
        if bucket is None:
            bucket = self.S3_BUCKET
        try:
            result = self.getClient().get_object(bucket, path)
            get_dagster_logger().info(f"file {result.status}")
            return result.data
        except Exception as ex:
            get_dagster_logger().info(
                f"file {path} not found in {bucket} at {self.S3_ADDRESS} {ex}"
            )
            raise Exception(
                f"file {path} not found in {bucket} at {self.S3_ADDRESS} {ex}"
            )

    def downloadFile(self, path=None, bucket="test", filename=None):
        """Download file from S3 to local filesystem."""
        if bucket is None:
            bucket = self.S3_BUCKET
        try:
            result = self.getClient().fget_object(bucket, path, file_path=filename)
            get_dagster_logger().info(f"file {filename}")
            return filename
        except Exception as ex:
            get_dagster_logger().error(
                f"file {path} not found in {bucket} at {self.S3_ADDRESS} {ex}"
            )
            raise Exception(
                f"file {path} not found in {bucket} at {self.S3_ADDRESS} {ex}"
            )

    def get_stream(self, path="test", bucket=None):
        """Get a file as a stream object from S3/MinIO without loading entire content into memory."""
        if bucket is None:
            bucket = self.S3_BUCKET
        try:
            result = self.getClient().get_object(bucket, path)
            get_dagster_logger().info(
                f"opened stream for file {path} with status {result.status}"
            )
            return result
        except Exception as ex:
            get_dagster_logger().info(
                f"file {path} not found in {bucket} at {self.S3_ADDRESS} {ex}"
            )
            raise Exception(
                f"file {path} not found in {bucket} at {self.S3_ADDRESS} {ex}"
            )

    def putFile_text(
        self,
        data,
        metadata={},
        path="test",
        content_type="text/plain",
        bucket=None,
    ):
        """Upload text data to S3."""
        if bucket is None:
            bucket = self.S3_BUCKET
        try:
            result = self.getClient().put_object(
                bucket,
                path,
                data=io.BytesIO(data.encode("utf-8")),
                length=len(data),
                content_type=content_type,
                metadata=metadata,
            )
            get_dagster_logger().info(
                "created {0} object; etag: {1}, version-id: {2}".format(
                    result.object_name, result.etag, result.version_id
                )
            )
            get_dagster_logger().info(f"file {result.object_name}")
            return result.object_name
        except Exception as ex:
            get_dagster_logger().info(
                f"file {path} failed to push to {bucket} at {self.S3_ADDRESS} {ex}"
            )
            raise Exception(
                f"file {path} failed to push to {bucket} at {self.S3_ADDRESS} {ex}"
            )

    def putFile(
        self,
        data,
        metadata={},
        path="test",
        content_type="application/octet-stream",
        bucket=None,
    ):
        """Upload binary data to S3."""
        if bucket is None:
            bucket = self.S3_BUCKET
        try:
            result = self.getClient().put_object(
                bucket,
                path,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type,
                metadata=metadata,
            )
            get_dagster_logger().info(
                "created {0} object; etag: {1}, version-id: {2}".format(
                    result.object_name, result.etag, result.version_id
                )
            )
            get_dagster_logger().info(f"file {result.object_name}")
            return result.object_name
        except Exception as ex:
            get_dagster_logger().info(
                f"file {path} failed to push to {bucket} at {self.S3_ADDRESS} {ex}"
            )
            raise Exception(
                f"file {path} failed to push to {bucket} at {self.S3_ADDRESS} {ex}"
            )

    def putStream(
        self,
        stream,
        length=-1,
        metadata={},
        path="test",
        content_type="application/octet-stream",
        bucket=None,
    ):
        """Upload data from a stream object directly to S3/MinIO."""
        if bucket is None:
            bucket = self.S3_BUCKET
        try:
            result = self.getClient().put_object(
                bucket,
                path,
                data=stream,
                length=length,
                content_type=content_type,
                metadata=metadata,
            )
            get_dagster_logger().info(
                "created {0} object; etag: {1}, version-id: {2}".format(
                    result.object_name, result.etag, result.version_id
                )
            )
            get_dagster_logger().info(f"file {result.object_name}")
            return result.object_name
        except Exception as ex:
            get_dagster_logger().info(
                f"file {path} failed to push to {bucket} at {self.S3_ADDRESS} {ex}"
            )
            raise Exception(
                f"file {path} failed to push to {bucket} at {self.S3_ADDRESS} {ex}"
            )
