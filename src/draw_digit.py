import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

from src.clustering import predict_knn

CANVAS_SIZE = 900
BRUSH_WIDTH = 20

def preprocess_image(image: Image.Image) -> torch.Tensor | None:
    image  = image.convert("L")
    bounds = image.getbbox()
    if bounds is None:
        return None

    digit = image.crop(bounds)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    width, height = digit.size
    pixels = torch.frombuffer(bytearray(digit.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(height, width).float()

    # Center the brightness mass, keeping the entire digit inside the image.
    center_x = (pixels.sum(dim=0) * torch.arange(width)).sum() / pixels.sum()
    center_y = (pixels.sum(dim=1) * torch.arange(height)).sum() / pixels.sum()
    left = max(0, min(28 - width,  round(13.5 - center_x.item())))
    top  = max(0, min(28 - height, round(13.5 - center_y.item())))

    prepared = Image.new("L", (28, 28), color=0)
    prepared.paste(digit, (left, top))
    tensor = torch.frombuffer(bytearray(prepared.tobytes()), dtype=torch.uint8)
    return tensor.reshape(1, 1, 28, 28).float() / 255.0


def predict_digit(
    model:  nn.Module,
    image:  torch.Tensor,
    device: torch.device,
    knn_embeddings: torch.Tensor,
    knn_labels: torch.Tensor,
    neighbors: int,
) -> tuple[int, float, int]:
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(image.to(device))
        probabilities = model.classifier(embedding).softmax(dim=1)
        confidence, digit = probabilities[0].max(dim=0)
        knn_digit = predict_knn(embedding, knn_embeddings, knn_labels, neighbors)
    return digit.item(), confidence.item(), knn_digit[0].item()


def create_gui(
    model: nn.Module,
    device: torch.device,
    knn_embeddings: torch.Tensor,
    knn_labels: torch.Tensor,
    neighbors: int,
) -> tk.Tk:
    root = tk.Tk()
    root.title("MNIST digit recognition")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", highlightthickness=0)
    canvas.pack(padx=12, pady=12)

    # Mirror canvas strokes into Pillow: no screenshots or screen coordinates needed.
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
    drawing = ImageDraw.Draw(image)
    last_point = None
    result = tk.StringVar(value="Draw a digit, then press Enter")

    def draw(event: tk.Event) -> None:
        nonlocal last_point
        point = (event.x, event.y)
        start = last_point if last_point is not None else point
        canvas.create_line(*start, *point, fill="white", width=BRUSH_WIDTH, capstyle=tk.ROUND)
        drawing.line([start, point], fill=255, width=BRUSH_WIDTH)

        # Pillow lines need explicit round endpoints, including a single click.
        radius = BRUSH_WIDTH / 2
        for x, y in (start, point):
            bounds = (x - radius, y - radius, x + radius, y + radius)
            canvas.create_oval(*bounds, fill="white", outline="white")
            drawing.ellipse(bounds, fill=255)
        last_point = point
        result.set("Press Enter to predict")

    def stop_drawing(event: tk.Event) -> None:
        nonlocal last_point
        last_point = None

    # Button callbacks take no event; keyboard bindings pass a tkinter event.
    def clear(event: tk.Event | None = None) -> None:
        nonlocal last_point
        canvas.delete("all")
        drawing.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
        last_point = None
        result.set("Draw a digit, then press Enter")

    def predict(event: tk.Event | None = None) -> None:
        tensor = preprocess_image(image)
        if tensor is None:
            result.set("Draw a digit first")
            return
        digit, confidence, knn_digit = predict_digit(
            model, tensor, device, knn_embeddings, knn_labels, neighbors
        )
        result.set(f"Classifier: {digit}    Confidence: {confidence:.2%}\nkNN: {knn_digit}")

    buttons = tk.Frame(root)
    buttons.pack()
    tk.Button(buttons, text="Predict (Enter)", command=predict).pack(side=tk.LEFT, padx=6)
    tk.Button(buttons, text="Clear (C)", command=clear).pack(side=tk.LEFT, padx=6)
    tk.Label(root, textvariable=result, width=48, height=2).pack(padx=12, pady=12)

    canvas.bind("<ButtonPress-1>", draw)
    canvas.bind("<B1-Motion>", draw)
    root.bind("<ButtonRelease-1>", stop_drawing)
    root.bind("<Return>", predict)
    root.bind("<Key-c>", clear)
    root.bind("<Key-C>", clear)
    root.bind("<Key-q>", lambda event: root.destroy())
    root.bind("<Key-Q>", lambda event: root.destroy())
    canvas.focus_set()

    return root
