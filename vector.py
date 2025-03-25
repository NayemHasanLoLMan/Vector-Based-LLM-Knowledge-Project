import pymongo 
import requests

client = pymongo.MongoClient("mongodb+srv://hasanmahmudnayeem3027:JT8J4TYqlCLTVgJu@cluster0.qthig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.sample_mflix 
collection = db.movies


hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

items = collection.find().limit(5)
# for item in items:
#     print(item)

def genarate_embedding(text: str,) -> list[float]:

    resposense = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"}, 
        json={"inputs": text})
    
    if resposense.status_code != 200:
        raise ValueError(f"Request failed with status code {resposense.status_code}: {resposense.text}")
    
    return resposense.json()

# for doc in collection.find({'plot': {"$exists": True}, 'plot_embedding_hf': {"$exists": False}}).limit(500):
#     try:
#         print(f"Processing document ID: {doc['_id']}")
#         doc['plot_embedding_hf'] = genarate_embedding(doc['plot'])
#         collection.replace_one({'_id': doc['_id']}, doc)
#     except Exception as e:
#         print(f"Failed to process document ID {doc['_id']}: {e}")

# for doc in collection.find({'plot':{"$exists": True}}).limit(1000):
#     doc['plot_embedding_hf'] = genarate_embedding(doc['plot'])
#     collection.replace_one({'_id': doc['_id']}, doc)


query = "imaginary characters from outer space at war"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": genarate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 150,
    "limit": 5,
    "index": "PlotSemanticSearch",
      }}
]);

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')


# Count documents with the 'plot_embedding_hf' field
count = collection.count_documents({'plot_embedding_hf': {'$exists': True}})
print(f"Number of documents with 'plot_embedding_hf' field: {count}")
