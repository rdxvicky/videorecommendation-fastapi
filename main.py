from fastapi import FastAPI, HTTPException
import torch
import pickle
from google.cloud import storage
import torch.nn as nn
from typing import List

app = FastAPI()

class RecSysModel(nn.Module):
    def __init__(self, num_users, num_videos, embedding_size=64):
        super(RecSysModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.video_embedding = nn.Embedding(num_videos, embedding_size)
        self.fc1 = nn.Linear(embedding_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
    
    def forward(self, user, video):
        user_emb = self.user_embedding(user)
        video_emb = self.video_embedding(video)
        x = user_emb * video_emb
        x = torch.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x.squeeze()

# Load model and mappings
storage_client = storage.Client()
bucket_name = 'pytorch-recommendation'
bucket = storage_client.bucket(bucket_name)

# Load the model
model_blob = bucket.blob('model_artifacts/model.pkl')
model_blob.download_to_filename('/tmp/model.pkl')
model = RecSysModel(num_users=10, num_videos=16)
model.load_state_dict(torch.load('/tmp/model.pkl'))
model.eval()

# Load the encoders
def load_pickle_from_gcs(filename):
    blob = bucket.blob(f'model_artifacts/{filename}')
    blob.download_to_filename(f'/tmp/{filename}')
    with open(f'/tmp/{filename}', 'rb') as f:
        return pickle.load(f)

user_to_user_encoded = load_pickle_from_gcs('user_to_user_encoded.pkl')
video_to_video_encoded = load_pickle_from_gcs('video_to_video_encoded.pkl')
user_encoded_to_user = load_pickle_from_gcs('user_encoded_to_user.pkl')
video_encoded_to_video = load_pickle_from_gcs('video_encoded_to_video.pkl')

@app.post("/predict")
async def predict(user_id: int, num_recommendations: int = 10):
    if user_id not in user_to_user_encoded:
        raise HTTPException(status_code=400, detail="User ID not found")
    
    user_idx = user_to_user_encoded[user_id]
    video_ids = list(video_to_video_encoded.keys())
    video_idxs = [video_to_video_encoded[vid] for vid in video_ids]
    
    user_array = torch.LongTensor([user_idx] * len(video_idxs))
    video_array = torch.LongTensor(video_idxs)
    
    with torch.no_grad():
        predictions = model(user_array, video_array)
    
    predicted_ratings = predictions.numpy()
    top_indices = predicted_ratings.argsort()[-num_recommendations:][::-1]
    recommended_video_ids = [video_encoded_to_video[idx] for idx in top_indices]
    
    return {"recommended_video_ids": recommended_video_ids}
