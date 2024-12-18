# mapping.py
import json
import os

mapping_file = "sign_names.json"

# Default mapping from class IDs to sign names
default_sign_names = {
    "0": "Ograniczenie prędkości (20km/h)",
    "1": "Ograniczenie prędkości (30km/h)",
    "2": "Ograniczenie prędkości (50km/h)",
    "3": "Ograniczenie prędkości (60km/h)",
    "4": "Ograniczenie prędkości (70km/h)",
    "5": "Ograniczenie prędkości (80km/h)",
    "6": "Koniec ograniczenia prędkości (80km/h)",
    "7": "Ograniczenie prędkości (100km/h)",
    "8": "Ograniczenie prędkości (120km/h)",
    "9": "Zakaz wyprzedzania",
    "10": "Zakaz wyprzedzania dla pojazdów powyżej 3.5t",
    "11": "Pierwszeństwo na najbliższym skrzyżowaniu",
    "12": "Droga z pierwszeństwem",
    "13": "Ustąp pierwszeństwa",
    "14": "Stop",
    "15": "Zakaz ruchu",
    "16": "Zakaz ruchu pojazdów powyżej 3.5t",
    "17": "Zakaz wjazdu",
    "18": "Ogólne niebezpieczeństwo",
    "19": "Niebezpieczny zakręt w lewo",
    "20": "Niebezpieczny zakręt w prawo",
    "21": "Niebezpieczne zakręty",
    "22": "Nierówna droga",
    "23": "Śliska nawierzchnia",
    "24": "Zwężenie jezdni z prawej strony",
    "25": "Roboty drogowe",
    "26": "Sygnalizacja świetlna",
    "27": "Przejście dla pieszych",
    "28": "Dzieci na drodze",
    "29": "Przejazd dla rowerzystów",
    "30": "Uwaga na oblodzenie",
    "31": "Zwierzęta dzikie",
    "32": "Koniec wszelkich ograniczeń",
    "33": "Nakaz jazdy w prawo",
    "34": "Nakaz jazdy w lewo",
    "35": "Nakaz jazdy prosto",
    "36": "Nakaz jazdy prosto lub w prawo",
    "37": "Nakaz jazdy prosto lub w lewo",
    "38": "Nakaz jazdy w prawo",
    "39": "Nakaz jazdy w lewo",
    "40": "Ruch okrężny",
    "41": "Koniec zakazu wyprzedzania",
    "42": "Koniec zakazu wyprzedzania przez pojazdy powyżej 3.5t",
}

# Load existing mapping or create a new one
if os.path.exists(mapping_file):
    with open(mapping_file, "r", encoding="utf-8") as f:
        sign_names = json.load(f)
else:
    sign_names = default_sign_names


def save_mapping():
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(sign_names, f, ensure_ascii=False, indent=4)
