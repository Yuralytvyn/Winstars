import os
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch

import sys
print(sys.executable)




def train_animal_ner(train_data, epochs=5, output_dir="models/animal_ner"):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label("ANIMAL")

    examples = []
    for text, ann in train_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, ann))

    nlp.initialize(lambda: examples)

    for epoch in range(epochs):
        random.shuffle(examples)
        losses = {}

        for batch in minibatch(examples, size=8):
            nlp.update(batch, drop=0.2, losses=losses)

        print(f"Epoch {epoch + 1}: {losses}")

    nlp.to_disk(output_dir)
    print(f"Saved to {output_dir}")


text = "I can see a spider at the picture, can't i?"
def find_animal(text):
    nlp = spacy.load("models/animal_ner")
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "ANIMAL":
            return ent.text.lower()

    return None



