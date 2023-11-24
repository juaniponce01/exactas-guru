import weaviate

auth_config = weaviate.AuthApiKey(api_key="YOUR-WEAVIATE-API-KEY")

client = weaviate.Client(
  url="https://rag-agent-cluster-wfzijl47.weaviate.network",
  auth_client_secret=auth_config
)