
# TransBind: Enhancing precise detection of DNA-binding proteins and DNA-protein binding residues using language model and deep learning



![TransBind Opening Diagram (2)](https://github.com/user-attachments/assets/0b948185-ae29-46d9-9ad6-779a0ac9e6b7)







## Abstract

We present TransBind, a highly accurate deep learning pipeline to identify DNA binding proteins and DNA binding amino acids within protein sequences by employing self-attention based transfer learning instead of using historical profile data for the first time. Global and local feature extractions by self-attention and inception learning, along with synthesized data generation for coping up with the data imbalance problem, allows TransBind to perform better than the previously proposed methods in most of the metrices. We report the performance of TransBind in comparison with other proposed methods on the benchmark datasets PDB1075, PDB186 for DNA binding protein identification, and PDNA41,PDNA224,PDNA316,PDNA543 for DNA binding amino acid residue identification. We also report that the running time is reduced significantly than any other previously reported methods while keeping a higher accuracy. 


## Tasks and Used Datasets

| Task Name                                   | Used Dataset                                |
| ------------------------------------------- | ------------------------------------------- |
| DNA Binding Protein Identification (Training)  | PDB-1075                                    |
| DNA Binding Protein Identification (Testing)   | PDB-186                                     |
| DNA Binding Amino Acid Identification (Training) | PDNA-224, PDNA-316, PDNA-543, DNA-573       |
| DNA Binding Amino Acid Identification (Testing)  | PDNA-41, DNA-129                            |
| RNA Binding Amino Acid Identification (Training) | RNA-495                                     |
| RNA Binding Amino Acid Identification (Testing)  | RNA-117   



## Dataset Link

All the used datasets are provided here:
[TransBind DATASET](https://drive.google.com/drive/folders/13dZsgurLKU8wR0YVdfzMX_GImqkqxgCW?usp=sharing)


For the latest dataset please download it from (Recommended) :
[Transbind Dataset Full](https://zenodo.org/records/10215073)


The Dataset contains the following data:

- DNA binding Protein datasets:
    - PDB 1075 (Raw data)
    - PDB 1075 (ProtTrans features)
    - PDB 186 (Raw data)
    - PDB 186 (ProtTrans features)
    - Trained Best model weights

- DNA binding Amino Acid datasets:
    - PDNA 224 (Raw data and ProtTrans Features)
    - PDNA 316 (Raw data and ProtTrans Features)
    - PDNA 543 (Raw data and ProtTrans Features)
    - PDNA 41 (Raw data and ProtTrans Features)
    - Trained Best model weights
- Used data to find the significance level with [iProDNA](https://pubmed.ncbi.nlm.nih.gov/31881828/) and Residue wise accuracy analysis
- The original ProtTrans-X-50 model weights to generate features from protein sequence.


The script to generate the protbert features from the raw sequences are provided in "generate_protbert_features.ipynb" notebook






