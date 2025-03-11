from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType


class MilvusHandler:
    def __init__(self):
        self.milvus_client = MilvusClient("./milvus_demo.db")
        self.collection_name = "alvin_collection"
        self.setup_collection()

    def setup_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="divide_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        schema = CollectionSchema(fields, "Collection for storing text + embeddings")

        if self.milvus_client.has_collection(collection_name=self.collection_name):
            print("collection is existing!")
        else: 
            self.milvus_client.create_collection(
                collection_name=self.collection_name, 
                schema=schema, 
                metric_type='IP'
            )

            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.milvus_client.release_collection(collection_name=self.collection_name)
            self.milvus_client.drop_index(
                collection_name=self.collection_name, index_name="vector"
            )
            index_params = self.milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="vector_index",
                index_type="FLAT", 
                metric_type="IP", 
                params={},
            )
            self.milvus_client.create_index(
                collection_name=self.collection_name, index_params=index_params, sync=True
            )

    def insert_data(self, vector_data):
        self.milvus_client.insert(
            collection_name=self.collection_name,
            data=vector_data
        )

    def search(self, query_vector, limit=1):
        search_params = {
            "metric_type": "IP",
            "params": {}
        }
        return self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text", "divide_text"],
            search_params=search_params
        )