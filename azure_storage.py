from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Ensure the required environment variables are set
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# Replace with your Azure Storage account details
  # Replace with your container name
LOCAL_FILE_PATH = 'converted_vocals.wav'
BLOB_NAME = 'converted_vocals.wav'  # This can be different if you want

# Step 1: Create BlobServiceClient
connection_str = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_str)

# Step 2: Upload file to blob storage
blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

with open(LOCAL_FILE_PATH, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)

print(f"✅ Uploaded {LOCAL_FILE_PATH} to blob storage.")

# Step 3: Generate SAS URL (valid for 1 hour)
sas_token = generate_blob_sas(
    account_name=STORAGE_ACCOUNT_NAME,
    container_name=CONTAINER_NAME,
    blob_name=BLOB_NAME,
    account_key=STORAGE_ACCOUNT_KEY,
    permission=BlobSasPermissions(read=True),
    expiry=datetime.utcnow() + timedelta(hours=1)
)

blob_url_with_sas = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{BLOB_NAME}?{sas_token}"

print("🔗 SAS URL for audio:", blob_url_with_sas)
