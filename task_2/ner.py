import os
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch

import sys
print(sys.executable)

TRAIN_DATA = [
("Is that a dog sitting in the grass?", {"entities":[(10,13,"ANIMAL")]}),
("I think I see a dog near the tree.", {"entities":[(16,19,"ANIMAL")]}),
("There is a small dog in the corner of the image.", {"entities":[(16,19,"ANIMAL")]}),
("Look at that dog running across the field!", {"entities":[(13,16,"ANIMAL")]}),

("A tall horse is standing in the field.", {"entities":[(7,12,"ANIMAL")]}),
("Can you notice the horse behind the fence?", {"entities":[(19,24,"ANIMAL")]}),
("Someone captured a beautiful horse in this photo.", {"entities":[(27,32,"ANIMAL")]}),
("That horse looks strong and fast.", {"entities":[(5,10,"ANIMAL")]}),

("I can clearly see an elephant walking there.", {"entities":[(18,26,"ANIMAL")]}),
("The elephant in the picture looks huge.", {"entities":[(4,12,"ANIMAL")]}),
("Wow, an elephant right in the middle of the frame!", {"entities":[(8,16,"ANIMAL")]}),
("Is that elephant drinking water?", {"entities":[(8,16,"ANIMAL")]}),

("What a beautiful butterfly flying here.", {"entities":[(17,26,"ANIMAL")]}),
("There is a bright butterfly on the flower.", {"entities":[(18,27,"ANIMAL")]}),
("I spotted a butterfly near the plant.", {"entities":[(11,20,"ANIMAL")]}),
("Do you see the butterfly above the leaves?", {"entities":[(15,24,"ANIMAL")]}),

("A chicken is walking across the yard.", {"entities":[(2,9,"ANIMAL")]}),
("That chicken looks funny in the photo.", {"entities":[(5,12,"ANIMAL")]}),
("Someone drew a chicken on the wall.", {"entities":[(16,23,"ANIMAL")]}),
("Is the chicken pecking the ground?", {"entities":[(7,14,"ANIMAL")]}),

("A sleepy cat is lying on the sofa.", {"entities":[(10,13,"ANIMAL")]}),
("Look at that fluffy cat in the basket.", {"entities":[(21,24,"ANIMAL")]}),
("I believe the cat is hiding under the table.", {"entities":[(14,17,"ANIMAL")]}),
("Such a cute cat staring at the camera.", {"entities":[(11,14,"ANIMAL")]}),

("There is a cow grazing in the meadow.", {"entities":[(11,14,"ANIMAL")]}),
("That cow seems calm and relaxed.", {"entities":[(5,8,"ANIMAL")]}),
("A cow appears near the barn.", {"entities":[(2,5,"ANIMAL")]}),
("Do you notice the cow behind the farmer?", {"entities":[(18,21,"ANIMAL")]}),

("A goat is climbing the rocks.", {"entities":[(2,6,"ANIMAL")]}),
("Look at that mountain goat!", {"entities":[(21,25,"ANIMAL")]}),
("The goat stands proudly on the hill.", {"entities":[(4,8,"ANIMAL")]}),
("Someone photographed a goat today.", {"entities":[(23,27,"ANIMAL")]}),

("There is a spider on the wall.", {"entities":[(11,17,"ANIMAL")]}),
("What a scary spider in the corner!", {"entities":[(12,18,"ANIMAL")]}),
("I noticed a spider crawling slowly.", {"entities":[(12,18,"ANIMAL")]}),
("That spider looks huge!", {"entities":[(5,11,"ANIMAL")]}),

("A squirrel is eating a nut.", {"entities":[(2,10,"ANIMAL")]}),
("Look at the squirrel on the branch.", {"entities":[(12,20,"ANIMAL")]}),
("Someone spotted a squirrel in the park.", {"entities":[(17,25,"ANIMAL")]}),
("The squirrel runs quickly across the road.", {"entities":[(4,12,"ANIMAL")]}),
("Do you see a dog anywhere in this image?", {"entities":[(13,16,"ANIMAL")]}),
("Is a dog present somewhere in the picture?", {"entities":[(5,8,"ANIMAL")]}),
("Could there be a dog visible in this photo?", {"entities":[(17,20,"ANIMAL")]}),

("Can you notice a horse in the scene?", {"entities":[(17,22,"ANIMAL")]}),
("Is a horse standing somewhere in this picture?", {"entities":[(5,10,"ANIMAL")]}),
("Do we have a horse appearing in the image?", {"entities":[(12,17,"ANIMAL")]}),

("Do you detect an elephant in this photo?", {"entities":[(17,25,"ANIMAL")]}),
("Is an elephant somewhere inside the picture?", {"entities":[(6,14,"ANIMAL")]}),
("Could you find an elephant in the scene?", {"entities":[(15,23,"ANIMAL")]}),

("Do you see a butterfly flying in this image?", {"entities":[(13,22,"ANIMAL")]}),
("Is a butterfly visible somewhere here?", {"entities":[(5,14,"ANIMAL")]}),
("Can a butterfly be spotted in the picture?", {"entities":[(6,15,"ANIMAL")]}),

("Do you notice a chicken in this scene?", {"entities":[(17,24,"ANIMAL")]}),
("Is a chicken present in the photograph?", {"entities":[(5,12,"ANIMAL")]}),
("Could there be a chicken somewhere here?", {"entities":[(17,24,"ANIMAL")]}),

("Do you see a cat anywhere in this photo?", {"entities":[(13,16,"ANIMAL")]}),
("Is a cat visible in the picture?", {"entities":[(5,8,"ANIMAL")]}),
("Could you spot a cat in this image?", {"entities":[(16,19,"ANIMAL")]}),

("Do you notice a cow somewhere here?", {"entities":[(17,20,"ANIMAL")]}),
("Is a cow present in the scene?", {"entities":[(5,8,"ANIMAL")]}),
("Could there be a cow inside this image?", {"entities":[(17,20,"ANIMAL")]}),

("Do you detect a sheep in the picture?", {"entities":[(15,20,"ANIMAL")]}),
("Is a sheep visible somewhere here?", {"entities":[(5,10,"ANIMAL")]}),
("Could you find a sheep in this image?", {"entities":[(17,22,"ANIMAL")]}),

("Do you see a spider anywhere here?", {"entities":[(13,19,"ANIMAL")]}),
("Is a spider present in this picture?", {"entities":[(5,11,"ANIMAL")]}),
("Could a spider be visible in the image?", {"entities":[(8,14,"ANIMAL")]}),

("Do you notice a squirrel in the scene?", {"entities":[(17,25,"ANIMAL")]}),
("Is a squirrel present in this picture?", {"entities":[(5,13,"ANIMAL")]}),
("Could there be a squirrel somewhere here?", {"entities":[(17,25,"ANIMAL")]}),

]


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

# train_animal_ner(TRAIN_DATA)
print(find_animal(text))
