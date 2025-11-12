import streamlit as st
import openai
import json
from PIL import Image

# Load your OpenAI key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Load and save memory ---
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

# --- Streamlit UI ---
st.title("🧠 AI Home Assistant Demo")
st.write("This assistant can remember what you tell it, analyze images, and talk to you like a household AI.")

uploaded_image = st.file_uploader("📸 Upload a picture (optional):", type=["jpg", "png", "jpeg"])
user_text = st.text_input("🗣️ Say or type something to your assistant:")

if st.button("Send"):
    if user_text:
        # Remember "I like" statements
        if "i like" in user_text.lower() or "my wife likes" in user_text.lower():
            fact = f"{user_text.strip('. ')}."
            if fact not in memory["facts"]:
                memory["facts"].append(fact)
                save_memory(memory)
                st.success("💾 Remembered that!")

        # Detect uploaded image
        image_desc = "No image provided."
        if uploaded_image:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded scene", use_container_width=True)
            image_desc = "User uploaded a photo."

        # Build context
        context = " | ".join(memory["facts"]) or "No stored memories yet"
        prompt = f"You currently remember: {context}. {image_desc} User said: {user_text}"

        # Ask GPT
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a friendly household assistant that remembers simple facts and comments on images."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            reply = response.choices[0].message.content
        st.write("**Assistant:**", reply)
