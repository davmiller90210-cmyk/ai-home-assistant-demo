import os, json, cv2
import openai
import pyttsx3
import speech_recognition as sr
from ultralytics import YOLO
from dotenv import load_dotenv

# ---------- SETUP ----------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

engine = pyttsx3.init()
recognizer = sr.Recognizer()
model = YOLO("yolov8n.pt")   # light YOLOv8 model

# ---------- MEMORY ----------
def load_memory():
    try:
        with open("memory.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"facts": []}

def save_memory(memory):
    with open("memory.json", "w") as f:
        json.dump(memory, f, indent=2)

memory = load_memory()

# ---------- HELPERS ----------
def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

def listen_or_type():
    try:
        with sr.Microphone() as source:
            print("🎤 Speak now (or wait 5 s to type):")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        return recognizer.recognize_google(audio)
    except Exception:
        return input("⌨️ Type your message: ")

def ask_gpt(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly household assistant that remembers simple facts from previous chats."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    print("Press Q in the camera window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("Mini Vision Assistant (press Q to quit)", annotated)

        user_text = listen_or_type()
        if not user_text:
            continue
        if user_text.lower() in ["quit", "exit", "stop"]:
            break

        # --- simple memory capture ---
        lower = user_text.lower()
        if "i like" in lower or "my wife likes" in lower:
            fact = f"{user_text.strip('. ')}."
            if fact not in memory["facts"]:
                memory["facts"].append(fact)
                save_memory(memory)
                print("💾 Remembered that!")

        # --- build context for GPT ---
        visible = [results[0].names[int(box.cls)] for box in results[0].boxes]
        seen = ", ".join(set(visible)) if visible else "nothing obvious"
        context = " | ".join(memory["facts"]) or "no stored memories yet"
        prompt = f"You currently remember: {context}. You see {seen}. User said: {user_text}"

        reply = ask_gpt(prompt)
        speak(reply)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- RUN ----------
if __name__ == "__main__":
    main()
