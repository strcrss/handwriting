import torch
from PIL import Image
import torchvision.transforms as transforms
from cnn import CNN
import matplotlib.pyplot as plt


def load_model(model_path = "results/cnn.pth"):
    model = CNN()
    model.load_state_dict(
        torch.load(
            "results/cnn_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    return model

def preprocess_image(image):
    image = Image.eval(image, lambda x: 255 - x)
    bbox = image.getbbox()

    if bbox is None:
        raise ValueError("画像に数字が見つかりません")

    image = image.crop(bbox)
    image.thumbnail((20, 20))
    new_image = Image.new("L", (28, 28), 0)
    x = (28 - image.width)//2
    y = (28 - image.height)//2
    new_image.paste(image, (x, y))
    new_image = Image.eval(new_image, lambda x: min(255, x * 2))

    return transforms.ToTensor()(new_image)
         

def predict_digit(image, model):
    image = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        probabilities = torch.softmax(outputs, dim = 1)
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    prediction = top_indices[0].tolist()
    confidence = (top_probabilities[0]*100).tolist()

    return prediction, confidence


def main():
    image = Image.open("8.png").convert("L")
    model = load_model()

    prediction, confidence = predict_digit(image, model)

    print("予測した数字:", prediction)
    print("確信度:", confidence)


if __name__ =="__main__":
    main()
