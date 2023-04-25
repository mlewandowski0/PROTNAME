import requests
from requests.adapters import HTTPAdapter, Retry
import re

re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_next_link(headers):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers)


def preprocessing_identity(x):
    return x

def within_length(x, max_length=510):
    val = x.split("\t")
    # x[0] = sequence, x[1] = protein_name, x[2] = length
    if int(val[2])  <= max_length:
        return x 
    return None

def get_from_uniprot(reviewed = None, annotation_score=None, format="tsv", data_from_organism = None, 
                     filepath_to_save = "dataset/data", fields=["sequence","protein_name","length"],
                       max_results=-1, preprocess_function=within_length, append=True):
    
    filepath_to_save = filepath_to_save + "." + format
    q_params = []

    if reviewed == True:
        q_params.append("reviewed:true")
    elif reviewed == False:
        q_params.append("reviewed:false")

    if data_from_organism != None:
        q_params.append(f"organism_id:{data_from_organism}")

    if annotation_score is not None:
        q_params.append(f"annotation_score:{annotation_score}")
    
    url = f"https://rest.uniprot.org/uniprotkb/search?query={'+AND+'.join(q_params)}&format={format}&size=500&fields={','.join(fields)}"
    
    flag = "w"
    if append == True:
        flag = "a"

    print(url)
    with open(filepath_to_save, flag) as f:
        if flag == "w":
            f.write("\t".join(fields) + "\n")
        i = 0
        buf = []
        for batch, total in get_batch(url):
            if max_results == -1:
                max_results = total

            data = batch.text.splitlines()[1:]
            
            for line in data:
                # preprocess
                preprocessed = preprocess_function(line) 
                if preprocessed is not None:
                    buf.append(preprocessed)
            
            i += len(data)
            print(f'{i} / {max_results}')

            if i >= int(max_results):
                break
        f.write("\n".join(buf))