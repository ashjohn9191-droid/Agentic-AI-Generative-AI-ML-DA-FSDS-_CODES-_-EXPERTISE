import streamlit as st
from nltk.chat.util import Chat, reflections

# =========================
# CHATBOT LOGIC (UNCHANGED)
# =========================
pairs = [
    [r"(.*)my name is (.*)", ["Hello %2, how are you today? 😊"]],
    [r"(.*)help(.*)", ["I can help you 👍"]],
    [r"(.*) your name ?", ["My name is thecleverprogrammer, but you can call me Robot 🤖"]],
    [r"how are you (.*)", ["I'm doing very well!", "I am great 😄"]],
    [r"sorry (.*)", ["It's alright", "No worries"]],
    [r"i'm (.*) (good|well|okay|ok)", ["Nice to hear that!", "Awesome 😎"]],
    [r"(hi|hey|hello|hola)(.*)", ["Hello 👋", "Hey there 😊"]],
    [r"what (.*) want ?", ["Make me an offer I can't refuse 😉"]],
    [r"(.*)created(.*)", ["Ashley created me using Python & NLTK 🚀"]],
    [r"(.*) (location|city) ?", ["Hyderabad, India 🇮🇳"]],
    [r"(.*)raining in (.*)", ["No rain in %2", "50% chance of rain in %2"]],
    [r"(.*)(sports|game|sport)(.*)", ["I'm a big fan of Cricket 🏏"]],
    [r"who (.*) (Cricketer|Batsman)?", ["Rohit Sharma 🏏"]],
    [r"(.*)faculaty name(.*)", ["My faculty name is Prakash Senapti"]],
    [r"quit", ["Bye! See you soon 👋"]],
    [r"(.*)", ["I'm still learning… can you rephrase? 🤔"]]
]

chatbot = Chat(pairs, reflections)

# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(page_title="NLP Chatbot", layout="centered")

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #000000, #1a0033, #2e1065);
    background-attachment: fixed;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #c084fc;
    margin-bottom: 30px;
}

/* USER INPUT BOX (BLUE) */
textarea {
    background-color: #0ea5e9 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-size: 16px !important;
}

/* BOT OUTPUT (PURPLE) */
.bot-box {
    background: #6d28d9;
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 16px;
    box-shadow: 0px 0px 15px rgba(109, 40, 217, 0.6);
}

/* SEND BUTTON */
.stButton>button {
    background-color: #facc15;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

.stButton>button:hover {
    background-color: #fde047;
}

</style>
""", unsafe_allow_html=True)

# =========================
# UI
# =========================
st.markdown('<div class="title">🤖 NLP Chatbot</div>', unsafe_allow_html=True)

user_input = st.text_area("💬 Type your message", height=100)

if st.button("🚀 Send"):
    if user_input.strip() != "":
        response = chatbot.respond(user_input.lower())
        st.markdown(f'<div class="bot-box">{response}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please type something!")

