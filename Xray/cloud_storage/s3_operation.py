import sys
import os
from Xray.exception import XRayException

class S3Operation:
    # Connecting the S3 bucket to push the data
    def sync_folder_to_S3(self, folder:str, bucket_name:str, bucket_folder_name:str) -> None:
        try:
            command:str = (
                f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}/"
            )
            os.system(command)

        except Exception as e:
            raise XRayException(e, sys)
        

    # Connecting the S3 bucket to fetch the data
    def sync_folder_from_S3(self, folder:str, bucket_name:str, bucket_folder_name:str) -> None:
        try:
            command:str = (
                f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder}"
            )
            os.system(command)

        except Exception as e:
            raise XRayException(e, sys)