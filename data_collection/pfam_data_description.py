import requests
import json
import os
from tqdm import tqdm
import argparse

def get_pfams_ids(path):
    data = []
    with open(pfam_family_file) as f:
        data = f.readlines()
    return [d.strip() for d in data]


def get_description(id):
    response = requests.get(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{id}")
    return response.json()

def get_description_mock(id):
    return {"metadata" : {"description" : "mock"}}

def collate_all_families(pfam_familiy_ids, mock=True):
    dataset = {}
    for pfam_id in tqdm(pfam_familiy_ids):
        if mock:
            dataset[pfam_id] = get_description_mock(pfam_id)
        else:
            dataset[pfam_id] = get_description(pfam_id)    

    return dataset



pfam_family_file = os.path.join("dataset", "relevant_pfam_ids.txt")
pfam_ids = get_pfams_ids(pfam_family_file)

with open(os.path.join("dataset", "pfam_data.json"), "w") as f:
    json.dump(collate_all_families(pfam_ids, mock=False), f)