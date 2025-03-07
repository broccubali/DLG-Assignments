import torch
from pykeen.pipeline import pipeline #sori I want namkeen
from pykeen.datasets import Hetionet
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
import pandas as pd

def load_hetionet():
    dataset = Hetionet()
    training_triples = dataset.training
    validation_triples = dataset.validation
    testing_triples = dataset.testing
    return training_triples, validation_triples, testing_triples

def train_kge_model():
    result = pipeline(
        dataset=Hetionet(), 
        model=TransE,
        training_kwargs={
            "num_epochs": 100,
            "batch_size": 128,
        },
    )
    return result


def get_alternate_drugs(model, entity_to_id, disease_id, triples_factory, top_k=10):
    disease_embedding = model.entity_representations[0](
        torch.tensor([entity_to_id[disease_id]])
    ).detach().numpy()
    
    all_entities = list(entity_to_id.keys())
    all_embeddings = model.entity_representations[0](torch.tensor(list(entity_to_id.values()))).detach().numpy()
    similarity_scores = all_embeddings @ disease_embedding.T
    
    direct_drugs = set(triples_factory.mapped_triples[triples_factory.mapped_triples[:, 0] == entity_to_id[disease_id]][:, 2].tolist())
    
    sorted_indices = similarity_scores.argsort(axis=0)[::-1]
    recommended_drugs = [all_entities[i] for i in sorted_indices if i not in direct_drugs][:top_k]
    
    return recommended_drugs

def main(disease_id):
    print("Training TransE model...")
    model_result = train_kge_model()  # No need to load triples separately
    model = model_result.model
    entity_to_id = model_result.training.entity_to_id  # Get entity mapping
    
    alternate_drugs = get_alternate_drugs(model, entity_to_id, disease_id, model_result.training)
    print(f"Recommended alternate drugs for disease {disease_id}: {alternate_drugs}")


# random disease
disease_id = "Disease:DOID:9352"
main(disease_id)
