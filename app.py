import tkinter as tk
from PIL import Image, ImageDraw
from predict import load_model, predict_digit
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手書き数字識別")
        self.root.geometry("800x400")
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10)

        self.model = load_model()

        self.strokes = []
        self.current_stroke = None

        self.canvas = tk.Canvas(
            left_frame,
            width=280,
            height=280,
            bg="black",
            cursor="cross"
        )
        self.canvas.pack(pady=20)

        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw)

        predict_button = tk.Button(
            left_frame,
            text="認識",
            command=self.predict
        )
        predict_button.pack(pady=5)

        clear_button = tk.Button(
            left_frame,
            text="消去",
            command=self.clear
        )
        clear_button.pack()

        self.result_label = tk.Label(
            right_frame,
            text="ここに認識結果が表示されます",
            font=("Arial", 14)
        )
        self.result_label.pack(pady=10)
        self.figure = Figure(figsize=(4.5, 2.5), dpi = 100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Prediction Probabilities")
        self.ax.set_xlabel("Digit")
        self.ax.set_ylabel("Probability (%)")
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 100)
        self.graph_canvas = FigureCanvasTkAgg(
            self.figure,
            master=right_frame
        )
        self.graph_canvas.get_tk_widget().pack(pady=10)
        self.graph_canvas.draw()



    def start_stroke(self, event):
        self.current_stroke = [(event.x, event.y)]
        self.strokes.append(self.current_stroke)


    def draw(self, event):
        x, y = event.x, event.y
        previous_x, previous_y = self.current_stroke[-1]

        self.canvas.create_line(
            previous_x, previous_y, x, y,
            fill="white",
            width=16,
            capstyle=tk.ROUND,
            smooth=True
        )

        self.current_stroke.append((x, y))

    def get_image(self):
        image = Image.new("L", (280, 280), 255)
        draw = ImageDraw.Draw(image)

        for stroke in self.strokes:
            if len(stroke) == 1:
                x, y = stroke[0]
                r = 8
                draw.ellipse(
                    (x-r, y-r, x+r, y+r),
                    fill=0
                )
            else:
                draw.line(
                    stroke,
                    fill=0,
                    width=16,
                    joint="curve"
                )
                r=8
                for x, y in [stroke[0], stroke[-1]]:
                    draw.ellipse(
                        (x-r, y-r, x+r, y+r),
                        fill=0
                    )
        return image

    def predict(self):
        try:
            image = self.get_image()
            confidences = predict_digit(image, self.model)

            top3 = sorted(
                enumerate(confidences),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            result = "認識結果\n"

            for i, (digit, confidence) in enumerate(top3):
                result += (
                    f"第{i+1}候補: {digit}"
                    f"({confidence:.2f}%)\n"
                )
            self.result_label.config(text=result)

            self.ax.clear()
            self.ax.bar(range(10), confidences)
            self.ax.set_title("Prediction Probabilities")
            self.ax.set_xlabel("Digit")
            self.ax.set_ylabel("Probability (%)")
            self.ax.set_xticks(range(10))
            self.ax.set_ylim(0, 100)
            self.figure.tight_layout()
            self.graph_canvas.draw()

        except ValueError as e:
            self.result_label.config(text=str(e))

    def clear(self):
        self.canvas.delete("all")
        self.strokes = []
        self.current_stroke = None
        self.result_label.config(
            text="ここに認識結果が表示されます"
        )



if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()




