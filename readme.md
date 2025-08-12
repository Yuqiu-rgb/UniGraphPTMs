# 			**UniGraphPTMs**

**1.Catalog description**

​	data: Data storage

​	src: Main code storage

​	model/embedding: Weight storage

------

**2.Requirements**

```
pandas

scikit-learn

transformers

matplotlib

umap-learn
```

Please download torch version>=2.0 or above to avoid version conflicts

------

**3.PLM model**

Please browse the required PLM official GitHub document to obtain the model embedding or weight.

We have provided the sample code of ProtT5 in the /src directory:：src/pretrained_embedding_generate.py

Please follow the official tutorials for the use of ESM-C and Saprot.

ESM-C:https://github.com/evolutionaryscale/esm

Saprot:https://github.com/westlake-repl/SaProtHub

------

**4.Onedrive description**

Due to the size limit of uploaded files in GitHub, we will upload embeddings and model weights to onedrive. Links can be found under each directory. Sometimes due to VPN issues, OneDrive will display abnormally. Please be patient and try a few times.

------

**5.How to Run**

Just run src/train.py to start training

```python
python train.py
```

Note: This version of the code may be slightly complex, but it is complete and functional. We are actively refactoring the code to provide researchers with a clearer understanding of the model internals. We will upload it as soon as the refactoring is complete. However, the currently released version does not affect the use of UniGraphPTMs.

------
**6.UniGraphPTMs-Mini**

To accommodate researchers with limited computing power or non-computer science backgrounds, we have also developed UniGraphPTMs-Mini. This version replaces BHGFN with the SaGCA module and optimizes overall parameters. The SaGCA module code can be found in the '/src' directory. Since the paper is still in the review stage and the relevant code has not been refactored, We will publish the codes and weights of all UniGraphPTMs-Mini in the near future. We are actively developing the UniGraphPTMs API to facilitate researchers’ use. However, this will take some time and we will upload it as soon as it is completed.


