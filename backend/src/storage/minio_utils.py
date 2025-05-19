import os
import aiobotocore.session

S3_ENDPOINT   = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET     = os.getenv("S3_BUCKET", "synth-data")

session = aiobotocore.session.get_session()

async def upload_fileobj(file_obj, key: str):
    async with session.create_client(
        "s3",
        endpoint_url=f"http://{S3_ENDPOINT}",
        aws_secret_access_key=S3_SECRET_KEY,
        aws_access_key_id=S3_ACCESS_KEY,
    ) as client:
        await client.put_object(Bucket=S3_BUCKET, Key=key, Body=file_obj)
    return key

async def download_fileobj(key: str):
    async with session.create_client(
        "s3",
        endpoint_url=f"http://{S3_ENDPOINT}",
        aws_secret_access_key=S3_SECRET_KEY,
        aws_access_key_id=S3_ACCESS_KEY,
    ) as client:
        resp = await client.get_object(Bucket=S3_BUCKET, Key=key)
        return await resp["Body"].read()
