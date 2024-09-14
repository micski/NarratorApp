import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget, QLineEdit, QLabel
from PyQt5.QtGui import QPixmap
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Inicjalizacja generatora tekstu z Hugging Face z użyciem CUDA (GPU)
text_generator = pipeline('text-generation', model='gpt2', device=0)  # 0 oznacza GPU

# Inicjalizacja generatora obrazów z Hugging Face (Stable Diffusion) z użyciem CUDA
image_generator = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
image_generator = image_generator.to("cuda")  # Użyj GPU do generacji obrazów

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Ustawienia głównego okna gry
        self.setWindowTitle('Gra Karciana z AI')
        self.setGeometry(100, 100, 800, 600)

        # Layout główny aplikacji (układ elementów w oknie)
        layout = QVBoxLayout()

        # Pole tekstowe do wyświetlania odpowiedzi AI
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)  # Pole tylko do odczytu
        layout.addWidget(self.output_text)  # Dodajemy pole tekstowe do layoutu

        # Pole do wprowadzania komend przez gracza
        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)  # Dodajemy pole do wprowadzania tekstu do layoutu

        # Przycisk do wysyłania komend gracza
        self.send_button = QPushButton('Wyślij Akcję', self)
        self.send_button.clicked.connect(self.handle_user_input)  # Po kliknięciu uruchamia funkcję handle_user_input
        layout.addWidget(self.send_button)  # Dodajemy przycisk do layoutu

        # Label do wyświetlania obrazów generowanych przez AI
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)  # Dodajemy label do layoutu

        # Ustawienie centralnego widgetu okna z layoutem
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def handle_user_input(self):
        # Pobieranie tekstu z pola wprowadzania
        user_input = self.input_field.text().lower().strip()
        # Obsługa akcji gracza
        if user_input in ["exit", "quit"]:
            self.close()
        elif user_input in ["atak", "obrona", "czar", "wycofaj się"]:
            # Generowanie odpowiedzi AI na akcję gracza
            ai_response = self.get_ai_response(f"Jako mistrz gry, jak odpowiedzieć na akcję gracza '{user_input}' w grze karcianej?")
            self.output_text.append(f"Gracz: {user_input}")
            self.output_text.append(f"AI: {ai_response}")
            # Generowanie obrazu na podstawie bardziej szczegółowego promptu
            image = self.generate_image(f"illustration of a fantasy card showing an action '{user_input}' with magical effects")
            self.show_image(image)
            self.input_field.clear()
        else:
            self.output_text.append("Nieznana akcja. Wybierz: 'atak', 'obrona', 'czar', 'wycofaj się'.")
            self.input_field.clear()

    def get_ai_response(self, user_input):
        # Generowanie odpowiedzi tekstowej z ustawieniami truncation i max_length
        response = text_generator(user_input, max_length=50, num_return_sequences=1, pad_token_id=50256, truncation=True, clean_up_tokenization_spaces=True)
        return response[0]['generated_text']

    def generate_image(self, prompt):
        # Generowanie obrazu na podstawie promptu
        image = image_generator(prompt).images[0]
        image.save("generated_image.png")  # Zapisz obraz, aby można było go załadować do UI
        return "generated_image.png"

    def show_image(self, image_path):
        # Ładowanie obrazu i wyświetlanie w UI
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)

def run_game():
    app = QApplication(sys.argv)  # Tworzenie instancji aplikacji
    window = GameWindow()  # Tworzenie okna gry
    window.show()  # Wyświetlanie okna gry
    sys.exit(app.exec_())  # Uruchomienie pętli aplikacji

if __name__ == "__main__":
    run_game()
