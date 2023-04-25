import os
from uniprot_collection import get_from_uniprot

get_from_uniprot( fields=["sequence","protein_name", "length"], append=False,
                 annotation_score=4, max_results=-1, 
                 filepath_to_save=os.path.join("dataset","proteins_large_annot4"))