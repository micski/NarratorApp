from transformers import pipeline

# Inicjalizacja generatora tekstu z Hugging Face. Używamy modelu GPT-2.
generator = pipeline('text-generation', model='gpt2')

# Funkcja do uzyskania odpowiedzi AI na podstawie wejścia użytkownika
def get_ai_response(user_input):
    # Generowanie odpowiedzi na podstawie tekstu wpisanego przez gracza
    response = generator(user_input, max_length=50, num_return_sequences=1, pad_token_id=50256)
    # Zwracamy wygenerowaną odpowiedź AI
    return response[0]['generated_text']

# Główna pętla gry, która będzie wykonywać się w konsoli
def main_game_loop():
    print("Witaj w grze karcianej z AI! Wybierz kartę lub podejmij decyzję.")
    print("Możliwe akcje: 'atak', 'obrona', 'czar', 'wycofaj się'. Wpisz 'exit' aby zakończyć grę.")
    while True:
        # Pobranie wejścia od użytkownika
        user_input = input("Twoja akcja: ").lower().strip()
        # Warunek zakończenia gry
        if user_input in ["exit", "quit"]:
            print("Dziękujemy za grę!")
            break
        elif user_input in ["atak", "obrona", "czar", "wycofaj się"]:
            # Uzyskanie odpowiedzi od AI
            ai_response = get_ai_response(f"Gracz wybrał akcję: {user_input}. Jak reaguje AI?")
            # Wyświetlenie odpowiedzi AI
            print("AI mówi:", ai_response)
        else:
            print("Nieznana akcja. Spróbuj ponownie.")

# Uruchomienie głównej pętli gry
if __name__ == "__main__":
    main_game_loop()
