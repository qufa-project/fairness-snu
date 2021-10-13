import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import umap.umap_ as umap

import boto3
import io
import json

config = None
with open('./config/config.json') as json_file:
    config = json.load(json_file)

client = boto3.client(
    's3',
    region_name=config['region_name'],
    aws_access_key_id=config['aws_access_key_id'],
    aws_secret_access_key=config['aws_secret_access_key']
)


def run_reduction(filename, seleced_dim):
    data = get_obj_from_s3(filename)
    dropped_data = data.drop(seleced_dim, axis=1)

    # t-SNE와 UMAP 알고리즘을 사용하여 2차원으로 차원 축소
    print("run tnse ========================")
    tsne_result = TSNE(n_components=2).fit_transform(dropped_data.values)
    print("tnse complete ========================")

    print("run umap ========================")
    umap_result = umap.UMAP().fit_transform(dropped_data.values)
    print("umap complete ========================")

    dimmed_data = data[seleced_dim].values

    reduced_data = pd.DataFrame({"tsne_1": tsne_result[:, 0], "tsne_2": tsne_result[:, 1],
                                 "umap_1": umap_result[:, 0], "umap_2": umap_result[:, 1],
                                 "sex": dimmed_data})

    filenameForUpload = removeExtensionFromFilename(filename=filename)
    uploadDfToS3(reduced_data, 'fairness/dimReduction/' +
                 filenameForUpload+'.csv')


def get_obj_from_s3(input_fname):
    obj = client.get_object(Bucket='qufa-test', Key=input_fname)
    data = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return data


def uploadDfToS3(data, key):
    with io.StringIO() as csv_buffer:
        data.to_csv(csv_buffer, index=False)

        response = client.put_object(
            Bucket='qufa-test', Key=key, Body=csv_buffer.getvalue()
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")


def removeExtensionFromFilename(filename):
    tokens = filename.split('.')
    tokens.pop()

    newFilename = ".".join(tokens)
    return newFilename
