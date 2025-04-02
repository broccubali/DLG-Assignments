# DLG-Assignments

## Part 1
### Here's what the question wants
BASICALLY, give alternative drigs for a given disease BUT
1. We can't have drugs directly connected to the disease in the KG
2. We need indirect drugs that could still be effective

So basically we wanna find new treamtents by finding drigs that aremn't directly related but could still be effectve

### What the code does
- Loads Hetionet- Gets the training, validation, and testing triples from the hetionwtr knowledge graph.
- Trains a TransE model - PyKEEN's pipeline to train a KGE model on herionet- embeddings gotta capture relationships
- Finds alternative drugs= so when you get a disease ID:
    - get the trained embedding of the disease
    - get similarity scores between the disease and all other entities (i used cosine)
    - kick out drugs directly connected to the disease (because we want alternatives, not knowns)
- return the top similar drug candidates
