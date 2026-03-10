import os
from PIL import Image

def main():

    os.makedirs("models", exist_ok=True)
    # NER
    if not os.path.exists("models/ner"):
        pass
        os.system("python ner.py")
    # CV
    if not os.path.exists("models/cv"):
        os.system("python cv.py")

    sentence = input()
    picture = "data/raw-img/elefante/e83cb10c28f5063ed1584d05fb1d4e9fe777ead218ac104497f5c978a4efbcb0_640.jpg"

    img = Image.open(picture)
    img.show()

if __name__=="__main__":
    main()