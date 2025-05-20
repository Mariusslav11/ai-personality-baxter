#!/usr/bin/env python3
##########################################
#                Imports                 #
##########################################

# Standard libraries for OS interaction, time, parsing, concurrency
import os
from dotenv import load_dotenv
import sys
import re          
import json
import time
import threading
import logging
import argparse
import subprocess
import queue
import asyncio
import aiohttp
import struct
from datetime import datetime, timedelta
import dateutil.parser
import re
import calendar
from pathlib import Path
from zoneinfo import ZoneInfo

# Controls GPT token limits
import tiktoken

# ROS2 - Python client library
import rclpy  
from rclpy.node import Node

import speech_recognition as sr  # Speech-to-text library
import pyttsx3  # Text-to-speech library
from playsound3 import playsound  # Fallback audio solution
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import pygame # Plays audio and syncs visuals

# Load environment variables
load_dotenv()

# OpenAI API
from openai import OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

from googleapiclient.discovery import build  # Google Search API
from transformers import pipeline  # Hugging Face summarization
import spacy # NLP summarization tokenizer

# Load spacy and HuggingFace pipelines
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ZERO_SHOT_CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Third party API keys
weather_api_key = os.getenv("WEATHER_API_KEY")
time_api_key = os.getenv("TIME_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# Default settings
DEFAULT_CITY = "Belfast"
DEFAULT_COUNTRY = "Northern Ireland"
DEFAULT_NEWS_SOURCE = "bbc-news"

# Tkinter & matplotlib for visualizer
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Audio monitoring and microphone processing
import numpy as np
import pyaudio  

CHUNK_SIZE = 2400 # Set audio chunk size
STREAMED_AUDIO_DATA = None  # Global variable for TTS streaming audio data ( for mouth visualization )

##########################################
#            Helper Functions            #
##########################################

def sanitize_snippet(snippet: str) -> str:
    
    # Removes or replaces unwanted phrases from the raw Google snippet.
    # Remove lines that start with "As of my last update..." or "As of my last knowledge..."
    snippet = re.sub(r"As of my last (knowledge|update).*?\.", "", snippet, flags=re.IGNORECASE)

    # Remove suggestions to check other sources for consistent replies
    snippet = snippet.replace("I recommend checking a reliable news source!", "")

    return snippet.strip()

##########################################
#        Factual Query Asynchronous      #
##########################################
async def get_weather_info_async(day="today"):

    # Asynchronously fetches current or forecast weather using WeatherAPI.
    # Defaults to 'today', but can handle forecasts for up to 10 days ahead.
    base_url = "http://api.weatherapi.com/v1/"
    if day.lower() == "today":
        url = f"{base_url}current.json?key={WEATHER_API_KEY}&q={DEFAULT_CITY}"
    else:
        url = f"{base_url}forecast.json?key={WEATHER_API_KEY}&q={DEFAULT_CITY}&days=10"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return "I couldn't retrieve the weather information at the moment."
            data = await response.json()
    if day.lower() == "today":
        temp_c = data['current']['temp_c']
        condition = data['current']['condition']['text']
        return f"The current weather in {DEFAULT_CITY} is {temp_c} degrees with {condition} skies."
    else:
        forecast_days = data.get('forecast', {}).get('forecastday', [])
        if not forecast_days:
            return "I couldn't retrieve the weather forecast data at the moment."
        requested_date = datetime.now().strftime("%Y-%m-%d")
        for forecast in forecast_days:
            if forecast['date'] == requested_date:
                day_temp = forecast['day']['avgtemp_c']
                day_condition = forecast['day']['condition']['text']
                return f"The forecast for {DEFAULT_CITY} on {requested_date} is {day_temp} degrees with {day_condition} skies."
        return f"Sorry, I couldn't find weather data for {day}. Please try a different day."

async def get_news_info_async(topic=None, source=DEFAULT_NEWS_SOURCE):

    # Fetches either top headlines or news about a specific topic from a specified source.
    base_url = "https://newsapi.org/v2/"
    if topic:
        url = f"{base_url}everything?q={topic}&sources={source}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=1"
    else:
        url = f"{base_url}top-headlines?sources={source}&apiKey={NEWS_API_KEY}&pageSize=1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return "I couldn't retrieve the news information at the moment."
            data = await response.json()
    articles = data.get('articles', [])
    if articles:
        top_article = articles[0]
        title = top_article.get('title', 'No title available')
        description = top_article.get('description', 'No description available')
        return f"Here's the latest news from {source.capitalize()}: {title}. {description}"
    else:
        return f"I couldn't find any news articles about {topic} from {source.capitalize()} at the moment."

async def get_wikipedia_summary_async(query):

    # Performs a Wikipedia search for the query and returns a summarized extract.
    # Falls back to the original extract if it's already short.
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "utf8": 1
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return "I couldn't retrieve the information from Wikipedia at the moment."
            data = await response.json()
    search_results = data.get("query", {}).get("search", [])
    if search_results:
        page_title = search_results[0]["title"]
        page_params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=page_params) as page_response:
                page_data = await page_response.json()
        page_id = next(iter(page_data['query']['pages']))
        page_extract = page_data['query']['pages'][page_id].get("extract", "No content found.")
        if len(page_extract) > 300:
            summarized = summarizer(page_extract, max_length=80, min_length=50, do_sample=False)
            summary = summarized[0]['summary_text']
            summary = ensure_complete_sentence(summary, page_extract)
        else:
            summary = page_extract
        if not summary.endswith(('.', '!', '?')):
            summary = summary.strip() + '.'
        return f"{page_title}: {summary}"
    return "I couldn't find any information on that topic."

def ensure_complete_sentence(summary, full_text):

    # Ensures that the summary ends in a complete sentence.
    # If the summary cuts off mid-sentence, it attempts to complete it using the full text.
    # (not functional)
    doc = nlp(full_text)
    sentences = [sent.text for sent in doc.sents]
    if not summary.endswith(('.', '!', '?')):
        for sent in sentences:
            if summary.endswith(sent[:len(summary)]):
                summary = sent
                break
    return summary

def google_search_sync(query, api_key=GOOGLE_API_KEY, cse_id=GOOGLE_CSE_ID):

    # Synchronous Google Custom Search wrapper.
    # Fetches a concise snippet for a given query.
    start = time.time()
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=1).execute()
    items = res.get('items', [])
    if items:
        return items[0].get('snippet', "No concise answer found.")
    return "No concise answer found."
    performance_logger.info(f"[Google Search] Snippet fetch time: {time.time() - start:.2f}s")

async def google_search_async(query, api_key=GOOGLE_API_KEY, cse_id=GOOGLE_CSE_ID):

    # Wraps the synchronous search call in an async executor thread
    # This allows non-blocking use inside the async workflow
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, google_search_sync, query, api_key, cse_id)
    return result

def handle_relative_date_query(user_input: str):

    # Detects natural language questions about dates
    # and returns the appropriate formatted date string
    now = datetime.now()
    user_input = user_input.lower()

    if "what's today's date" in user_input or "what's the current date" in user_input or "what day is it" in user_input:
        return now.strftime("%A, %B %d, %Y")

    # Check for "tomorrow"
    if "what's tomorrow's date" in user_input:
        return (now + timedelta(days=1)).strftime("%A, %B %d, %Y")

    # Check for "yesterday"
    if "what's yesterday's date" in user_input:
        return (now - timedelta(days=1)).strftime("%A, %B %d, %Y")

    # Check for "in X days/weeks"
    match = re.search(r"in (\d+) (day|days|week|weeks)", user_input)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        delta = timedelta(days=num) if "day" in unit else timedelta(weeks=num)
        return (now + delta).strftime("%A, %B %d, %Y")

    # Check for "next [weekday]"
    weekdays = list(calendar.day_name)
    for i, day in enumerate(weekdays):
        if f"next {day.lower()}" in user_input:
            days_ahead = (i - now.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).strftime("%A, %B %d, %Y")

        if f"last {day.lower()}" in user_input:
            days_behind = (now.weekday() - i + 7) % 7
            if days_behind == 0:
                days_behind += 7
            return (now - timedelta(days=days_behind)).strftime("%A, %B %d, %Y")

    return None

##########################################
#        GPT Rephrasing Helper           #
##########################################
def get_conversational_answer(user_input, snippet, personality_style):
    #Takes the user query and a snippet from Google,
    #then instructs GPT to rephrase it in a friendly, conversational style.

    # 1) Sanitize the snippet to remove disclaimers or "As of my last update..."
    cleaned_snippet = sanitize_snippet(snippet)

    # 2) Create a system prompt that forbids disclaimers
    system_prompt = f"""
You are Baxter, a {personality_style} AI assistant.
The user asked: "{user_input}"
The following concise information was retrieved from a real-time Google search:
"{cleaned_snippet}"

Please rephrase and expand on this information in a friendly, conversational style.
Do not mention knowledge cutoffs, training data, or anything about when the AI was last updated.
If the snippet contains time-sensitive information (like dates), assume it is up-to-date.
Never say "as of" or reference any past update dates like October 2023.
If the snippet seems incomplete, do your best to clarify or fill in relevant details, 
but do not invent contradictory facts.
Keep your response concise and limit it to a single paragraph.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}],
        max_tokens=150,
        n=1,
        temperature=0.7
    )
    return response.choices[0].message.content

def count_tokens(text, model="gpt-4o-mini"):
    
    # Counts the number of tokens used in a string for a given model using tiktoken
    # Used for avoiding truncation mid-sentence
    model_map = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-3.5": "gpt-3.5",
    }
    encoding_model = model_map.get(model, model)
    encoding = tiktoken.encoding_for_model(encoding_model)
    return len(encoding.encode(text))

def truncate_incomplete_sentence(text: str, max_tokens: int = 150) -> str:
    fallback = "You can find more information about this online!"
    model = "gpt-4o-mini"

    # Step 1: Combine all paragraphs into one
    combined_paragraph = " ".join([p.strip() for p in text.strip().split("\n\n")])

    # Step 2: Check if combined paragraph + fallback fits within max tokens
    candidate = f"{combined_paragraph} {fallback}"
    if count_tokens(candidate, model=model) <= max_tokens:
        return candidate

    # Step 3: If even combined paragraph is too long, try truncating at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', combined_paragraph)
    truncated_text = ""
    for sentence in sentences:
        potential_text = f"{truncated_text} {sentence}".strip()
        if count_tokens(f"{potential_text} {fallback}", model=model) > max_tokens:
            break
        truncated_text = potential_text

    # Step 4: If nothing fits, return fallback only
    if not truncated_text:
        return fallback

    # Step 5: Return truncated text + fallback
    return f"{truncated_text} {fallback}"

##########################################
#         Asynchronous Audio Player      #
##########################################
async def play_audio_async(file_path):

    # Plays an audio file asynchronously using pygame.
    # Ensures the rest of the system doesn't block while audio plays.
    loop = asyncio.get_event_loop()

    def blocking_audio_play():
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

    await loop.run_in_executor(None, blocking_audio_play)

##########################################
#       Custom Logging Filter & Setup    #
##########################################
class TerminalFilter(logging.Filter):

    # Filters terminal output to only show meaningful user interactions.
    # Keeps the logs from being too noisy in the GUI console.
    def filter(self, record):
        allowed_phrases = [
            "Listening for user input",
            "User said:",
            "Response:",
            "Could not understand the audio"
        ]
        message = record.getMessage()
        return any(message.startswith(phrase) for phrase in allowed_phrases)

# Setup for multiple logging destinations:
file_handler1 = logging.FileHandler("chatbot_node.log")
file_handler2 = logging.FileHandler("visualizer.log")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.addFilter(TerminalFilter())

# Performance metrics
performance_handler = logging.FileHandler("performance.log")
performance_handler.setLevel(logging.INFO)
performance_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

performance_logger = logging.getLogger("performance_logger")
performance_logger.setLevel(logging.INFO)
performance_logger.addHandler(performance_handler)
performance_logger.propagate = False

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler1, file_handler2, stream_handler]
)


# Global variable for the shared audio file path.
AUDIO_FILE_PATH = Path(__file__).parent / "response.mp3"
# Global variable for personality – default "friendly"
CURRENT_PERSONALITY = "friendly"

##########################################
#           Personality Code             #
##########################################
class Personality:

    # Managing class for different personalities
    # Reads from JSON config file (adjust location based to path on your machine)
    def __init__(self, personality_type, config_path="personalities.json"):
        config_path = "/home/marius/ros2_ws/src/baxter_pkg/baxter_pkg/personalities.json"
        with open(config_path) as file:
            self.personalities = json.load(file)
        self.set_personality(personality_type)

    def set_personality(self, personality_type):
        self.personality = self.personalities.get(personality_type, self.personalities["friendly"])
        global CURRENT_PERSONALITY
        CURRENT_PERSONALITY = personality_type

    def get_greeting(self):
        return self.personality["greeting"]

    def get_response_style(self):
        return self.personality["response_style"]

    def get_voice_model(self):
        return self.personality["voice_model"]

##########################################
#   GPTChatBotNode Class (Main Node)     #
##########################################
class GPTChatBotNode(Node):
    def __init__(self):
        logging.debug("Initializing GPTChatBotNode...")
        super().__init__('gpt_chatbot_node')

        # Thread safety
        self.lock = threading.Lock() 
        self.stop_flag = threading.Event()

        # Audio playback + mic listening threads
        self.audio_thread = None
        self.listen_thread = None

        # Prevents listening while greeting is playing
        self.greeting_done_event = threading.Event()
        self.is_playing = False

        # Zero-shot classifier used for factual vs conversational query detection
        self.zero_shot_classifier = ZERO_SHOT_CLASSIFIER
        
        # Initialize Pygame mixer for audio playback
        try:
            pygame.mixer.pre_init(frequency=24000, buffer=CHUNK_SIZE)
            pygame.init()
            pygame.mixer.init()
            logging.debug("Pygame mixer initialized successfully.")
        except pygame.error as e:
            logging.critical(f"Pygame mixer initialization failed: {e}")
            sys.exit(1)

        # Startup visual feedback, spinner while loading
        self.loading_complete = threading.Event()
        self.loading_thread = threading.Thread(target=self.show_loading)
        self.loading_thread.start()
        time.sleep(5)
        self.loading_complete.set()
        self.loading_thread.join()
        
        self.iteration_counter = 0 # Log rotation variabe (deletes last iteration after 3 iterations)
        self.recognizer = sr.Recognizer() # Speech-to-text engine
        self.tts_engine = pyttsx3.init() # Offline TTS (experimental engine if OpenAI doesnt work)
        self.personality_instance = Personality(personality_type="friendly")
        self.personality_changed = False
        self.startup_greeting_in_progress = True

        # Start listener thread. This runs forever in the background
        listen_thread = threading.Thread(target=self.listen_and_respond, daemon=True)
        listen_thread.start()

        # Used to pass messages between chatbot and user
        self.conversation_history = []

        # Direct google search when no OpenAI result returned
        self.google_api_key = GOOGLE_API_KEY
        self.google_cse_id = GOOGLE_CSE_ID
        self.trigger_search = False 

        # Attempt to detect user location/timezone
        try:
            import requests
            import tzlocal
            local_timezone = tzlocal.get_localzone()
            self.user_timezone = local_timezone.key
            response = requests.get("https://ipinfo.io/json")
            if response.status_code == 200:
                ip_data = response.json()
                city = ip_data.get("city", "Unknown location")
                country = ip_data.get("country", "")
                self.user_location = f"{city}, {country}"
            else:
                self.user_location = "Unknown location"
                self.user_timezone = "UTC"
        except Exception as e:
            logging.error(f"Error retrieving location and timezone: {e}")
            self.user_location = "Unknown location"
            self.user_timezone = "UTC"

        # Thread monitoring
        monitor_thread = threading.Thread(target=self.monitor_threads, daemon=True)
        monitor_thread.start()

        logging.debug("Constructor finished init. Now calling greet_user()...")
        self.greet_user()
        logging.debug("greet_user() completed. Node constructor done.")

        # Second listener for interruption control during audio playback
        stop_thread = threading.Thread(target=self.listen_for_stop, daemon=True)
        stop_thread.start()

    def show_loading(self):

        # Console animation during program loading
        animation = "|/-\\"
        idx = 0
        while not self.loading_complete.is_set():
            sys.stdout.write(f"\rLoading Baxter... {animation[idx % len(animation)]}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()

    def handle_exception(exc_type, exc_value, exc_traceback):

        # System wide exception handler
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    def reset_log_file(self):

        # Empties log file after 3 iterations
        try:
            log_path = "chatbot_node.log"
            with open(log_path, "w"):
                pass  # Truncate the file
            logging.info("chatbot_node.log has been reset after 3 iterations.")
        except Exception as e:
            logging.error(f"Failed to reset log file: {e}")

    def listen_and_respond(self):
        start_interaction = time.time()
        self.greeting_done_event.wait() # Wait until greeting stops playing
        while True:
            if self.startup_greeting_in_progress:
                continue # Dont listen while greeting is playing

            self.stop_flag.clear()

            # Record voice input
            with sr.Microphone() as source:
                logging.debug("Listening for user input")
                self.recognizer.adjust_for_ambient_noise(source)
                try:
                    start_stt = time.time()
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
                except sr.WaitTimeoutError:
                    continue

            try:
                start_time = time.time()
                user_input = self.recognizer.recognize_google(audio)
                end_stt = time.time()
                performance_logger.info(f"[STT] Duration: {end_stt - start_stt:.2f} seconds")
                logging.debug(f"User said: {user_input}")
                query_lower = user_input.lower()
                
                relative_date_answer = handle_relative_date_query(user_input)
                if relative_date_answer:
                    self.respond_with_voice(f"The date you're asking about is {relative_date_answer}.", self.personality_instance)
                    continue

                # Check for self-description queries
                if "who are you" in query_lower or "tell me about yourself" in query_lower:
                    response = self.personality_instance.personality.get("self_description", "")
                    print(f"[Personality Response] {response}")
                    self.respond_with_voice(response, self.personality_instance)
                    continue

                # Check for favourite colour
                if "favourite colour" in query_lower:
                    response = f"My favourite colour is {self.personality_instance.personality.get('favourite_colour', '')}"
                    print(f"[Personality Response] {response}")
                    self.respond_with_voice(response, self.personality_instance)
                    continue

                # Check for favourite book
                if "favourite book" in query_lower:
                    response = f"My favourite book is {self.personality_instance.personality.get('favourite_book', '')}"
                    print(f"[Personality Response] {response}")
                    self.respond_with_voice(response, self.personality_instance)
                    continue

                # Check for favourite food
                if "favourite food" in query_lower:
                    response = f"My favourite food is {self.personality_instance.personality.get('favourite_food', '')}"
                    print(f"[Personality Response] {response}")
                    self.respond_with_voice(response, self.personality_instance)
                    continue

                # Check for catchphrase
                if "catchphrase" in query_lower:
                    response = f"My catchphrase is: {self.personality_instance.personality.get('catchphrase', '')}"
                    print(f"[Personality Response] {response}")
                    self.respond_with_voice(response, self.personality_instance)
                    continue

                # "stop" and personality chane control
                if "stop" in user_input.lower():
                    logging.info("Stopping audio playback...")
                    print(f"Audio playback stopped")
                    self.stop_audio_playback()
                    continue
                if "change personality" in user_input.lower():
                    self.handle_voice_command(user_input, self.personality_instance)
                    continue

                # Determine if query is factual or not
                is_factual, entities = self.detect_factual_query(user_input)

                if is_factual:
                    # Factual queries:
                    start_fact = time.time()
                    if "weather" in user_input.lower():
                        day = "today"
                        for ent in entities:
                            if ent[1] == "DATE":
                                day = ent[0]
                                break
                        factual_response = asyncio.run(get_weather_info_async(day))

                    elif "news" in user_input.lower():
                        topic = None
                        for ent in entities:
                            if ent[1] in ["GPE", "ORG", "PERSON", "EVENT"]:
                                topic = ent[0]
                                break
                        factual_response = asyncio.run(get_news_info_async(topic=topic))

                    elif "time" in user_input.lower():
                        now = datetime.now()
                        hour = now.strftime("%I").lstrip("0")       # 12-hour format without leading zero
                        minute = now.minute
                        period = now.strftime("%p").lower()         # 'am' or 'pm'

                        if minute == 0:
                            current_time = f"exactly {hour} {period}"
                        else:
                            current_time = f"{minute} minute{'s' if minute != 1 else ''} past {hour} {period}"
                        factual_response = f"The current time is {current_time}."

                    else:
                        # For unknown factual queries, use Google search + GPT rephrasing
                        snippet = asyncio.run(google_search_async(user_input))
                        if snippet and snippet != "No concise answer found.":
                            factual_response = get_conversational_answer(user_input, snippet, CURRENT_PERSONALITY)
                        else:
                            factual_response = "I'm sorry, I couldn't find a direct answer to that question."

                    factual_response = truncate_incomplete_sentence(factual_response, max_tokens=150)
                    logging.debug(f"Response: {factual_response}")
                    self.respond_with_voice(factual_response, self.personality_instance)
                    performance_logger.info(f"[Factual] Query-to-Playback duration: {time.time() - start_fact:.2f} seconds")
                    continue  # skip normal GPT path

                else:
                    # Normal conversation -> GPT
                    response = self.get_gpt_response(user_input)
                    logging.debug(f"Response: {response}")
                    self.respond_with_voice(response, self.personality_instance)

                end_time = time.time()
                total_time = end_time - start_time
                performance_logger.info(f"Total response time: {total_time:.2f} seconds")
                performance_logger.info(f"[Interaction Loop] Total Duration: {time.time() - start_interaction:.2f} seconds")
                # Log file cleanup management
                self.iteration_counter += 1
                if self.iteration_counter >= 3:
                    self.reset_log_file()
                    self.iteration_counter = 0

            except sr.UnknownValueError:
                logging.debug("Could not understand the audio")
            except Exception as e:
                logging.error(f"Error: {e}")

    def greet_user(self):

        # Greeting function based on users host machine time
        logging.debug("greet_user() called!")
        now = datetime.now()
        if now.hour < 12:
            time_of_day = "morning"
        elif now.hour < 18:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        personality = CURRENT_PERSONALITY
        if personality == "friendly":
            greeting = f"Good {time_of_day}! My name is Baxter, how can I help?"
        elif personality == "evil":
            greeting = f"Salutations mortal, how may I be of service this {time_of_day}?"
        elif personality == "professional":
            greeting = f"Good {time_of_day}. How may I assist you today?"
        elif personality == "funny":
            greeting = f"Hey there! Ready for some fun? What can I do for you this {time_of_day}?"
        else:
            greeting = self.personality_instance.get_greeting()

        logging.debug(f"Response: {greeting}")
        self.respond_with_voice(greeting, self.personality_instance)

        if self.audio_thread is not None:
            logging.debug("Waiting for audio_thread to finish greeting...")
            self.audio_thread.join()
            logging.debug("audio_thread finished greeting.")

        self.startup_greeting_in_progress = False
        self.greeting_done_event.set() # Thread block from input capturing greeting
        logging.debug("greet_user() done, startup_greeting_in_progress set to False.")

    def detect_factual_query(self, query):
        
        #Returns (is_factual, entities).
        #'is_factual' is True if the question is likely a factual query.
        #Excludes self-referential queries like "who are you?"

        query_lower = query.lower().strip()
        self_referential = {"what are you", "who are you", "tell me about yourself"}
        if query_lower in self_referential:
            logging.debug("Query is self-referential; not factual.")
            return False, []

        # Hard-coded triggers for known factual domains
        factual_triggers = {"weather": 1.0, "time": 1.0, "news": 1.0}
        for keyword, score in factual_triggers.items():
            if keyword in query_lower:
                doc = nlp(query)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                logging.debug(f"Query contains '{keyword}'; using factual path.")
                return True, entities

        # Otherwise, do zero-shot classification
        candidate_labels = ["factual", "conversational"]
        result = self.zero_shot_classifier(query, candidate_labels)
        factual_score = result["scores"][result["labels"].index("factual")]
        is_factual = factual_score > 0.5

        doc = nlp(query)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logging.debug(f"Zero-shot classifier result: {result}, is_factual={is_factual}")
        return is_factual, entities

    def get_gpt_response(self, user_input):

        # Generates GPT response
        if self.stop_flag.is_set():
            logging.info("GPT response generation interrupted by user.")
            return "Response generation stopped."

        start_time = time.time()

        # System prompt defines the assistant's tone/personality
        system_message = {
            "role": "system",
            "content": self.personality_instance.personality["system_message"]
        }
        self.conversation_history.append({"role": "user", "content": user_input})
        conversation = [system_message] + self.conversation_history
        
        start_gpt = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            max_tokens=150,
            n=1,
            temperature=0.7
        )

        elapsed = time.time() - start_time  
        performance_logger.info(f"GPT response time: {elapsed:.2f} seconds")

        gpt_response = response.choices[0].message.content
        if gpt_response.lower().startswith("baxter: "):
            gpt_response = gpt_response[len("baxter: "):].strip()
        num_tokens = count_tokens(gpt_response, model="gpt-4o-mini")
        self.conversation_history.append({"role": "assistant", "content": gpt_response})
        end_gpt = time.time()
        performance_logger.info(f"[GPT] Generation Duration: {end_gpt - start_gpt:.2f} seconds")
        return gpt_response

    def respond_with_voice(self, response_text, personality_instance):
        start_tts = time.time()

        # If visualizer is present, let it know when TTS starts
        try:
            import __main__
            if hasattr(__main__, "visualizer"):
                __main__.visualizer.tts_start_time = start_tts
                __main__.visualizer.visualizer_sync_logged = False
        except Exception as e:
            logging.error(f"Error setting TTS start time in visualizer: {e}")

        # Clear neutral mouth override before playback starts
        try:
            import __main__
            if hasattr(__main__, "visualizer") and __main__.visualizer is not None:
                __main__.visualizer.force_neutral_mouth = False
        except Exception as e:
            logging.error(f"Error clearing neutral mouth: {e}")

        if self.stop_flag.is_set():
            logging.debug("Playback interrupted.")
            return
        try:
            # Use streaming TTS endpoint
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=self.personality_instance.get_voice_model(),
                input=response_text,
                response_format="pcm"  # Ensure proper format for real-time playback
            ) as response:
                self.is_playing = True
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=p.get_format_from_width(2),  # 16-bit PCM
                    channels=1,
                    rate=24000,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE
                )
                global STREAMED_AUDIO_DATA
                first_chunk = True
                for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                    if self.stop_flag.is_set():
                        logging.debug("Stop flag detected during TTS streaming.")
                        break
                    if first_chunk:
                        latency = time.time() - start_tts
                        performance_logger.info(f"[TTS] First Audio Chunk Latency: {latency:.2f} seconds")
                        first_chunk = False

                    # Write chunk to audio output
                    stream.write(chunk)
                    # Convert chunk bytes to a numpy array (waveform for visualizer)
                    try:
                        data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    except Exception as conv_e:
                        logging.error(f"Error converting chunk: {conv_e}")
                        data = np.zeros(CHUNK_SIZE, dtype=np.float32)
                    STREAMED_AUDIO_DATA = data  # Update global variable for visualization
                stream.stop_stream()
                performance_logger.info(f"[TTS] Total Playback Duration: {time.time() - start_tts:.2f} seconds")
                time.sleep(1) # Allow last audio chunk to finish playing avoiding cutoffs
                stream.close()
                p.terminate()
        except Exception as e:
            logging.error(f"Error during streaming audio playback: {e}")
        finally:
            self.is_playing = False

    def handle_voice_command(self, command, personality_instance):

        # Personality change voice handler
        if "change personality" in command:
            for personality in ["friendly", "professional", "funny", "evil"]:
                if personality in command:
                    self.personality_instance.set_personality(personality)
                    self.personality_changed = True
                    self.greet_user()
                    logging.info(f"Personality switched to: {personality}")
                    break

    def log_active_threads(self):

        # Active thread logging for debugging
        active_threads = threading.enumerate()
        logging.debug(f"Active Threads ({len(active_threads)}):")
        for thread in active_threads:
            logging.debug(f"Thread Name: {thread.name}, Alive: {thread.is_alive()}, Daemon: {thread.daemon}")

    def monitor_threads(self):

        # Periodically logs state of threads
        while True:
            self.log_active_threads()
            if not self.is_playing:
                logging.debug("Audio playback is not active.")
            time.sleep(5)

    def cleanup_threads(self):

        # Graceful shutdown thread cleanup
        threads_to_join = [self.audio_thread, self.listen_thread]
        for thread in threads_to_join:
            if thread and thread.is_alive():
                logging.debug(f"Waiting for thread {thread.name} to finish...")
                thread.join(timeout=2)
                if thread.is_alive():
                    logging.warning(f"Thread {thread.name} did not terminate properly.")
                else:
                    logging.debug(f"Thread {thread.name} terminated successfully.")

    def stop_audio_playback(self):

        # Interrupts audio playback, sets mouth to neutral, and resets visualizer.
        with self.lock:
            if not self.is_playing:
                logging.debug("No active playback to stop.")
                return
            self.stop_flag.set()
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            self.is_playing = False
            logging.info("Audio playback has been stopped.")

            global STREAMED_AUDIO_DATA
            STREAMED_AUDIO_DATA = np.zeros(CHUNK_SIZE, dtype=np.float32)

            # Reset mouth to neutral shape
            try:
                import __main__
                if hasattr(__main__, "visualizer") and __main__.visualizer is not None:
                    __main__.visualizer.set_neutral_mouth()
            except Exception as e:
                logging.error(f"Error resetting neutral mouth after stop: {e}")

    def listen_for_stop(self):

        # Dedicated mic thread that listens for 'stop' 
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                logging.debug("Stop listener active: Listening for 'stop' command...")
                while True:  # Run continuously
                    try:
                        audio = recognizer.listen(source, timeout=60, phrase_time_limit=2)
                        user_input = recognizer.recognize_google(audio).lower()
                        logging.debug(f"Stop listener heard: {user_input}")
                        if "stop" in user_input:
                            logging.info("Stop command detected.")
                            if self.is_playing:
                                self.stop_audio_playback()
                                self.stop = True

                    except sr.WaitTimeoutError:
                        logging.debug("Stop listener: No input detected, continuing...")
                    except sr.UnknownValueError:
                        logging.debug("Stop listener: Could not understand audio, retrying...")
        except Exception as e:
            logging.critical(f"Critical error in stop listener: {e}")

    def shutdown_node(self):

        # Full cleanup on program termination
        logging.info("Shutting down node and cleaning up resources.")
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            pygame.quit()
        except Exception as e:
            logging.error(f"Error during pygame cleanup: {e}")
        finally:
            self.destroy_node()
            rclpy.shutdown()
            # Attempt to shut down the visualizer
            try:
                import __main__
                if hasattr(__main__, "visualizer") and __main__.visualizer is not None:
                    __main__.visualizer.root.destroy()
            except Exception as e:
                logging.error(f"Error during visualizer shutdown: {e}")
            print("Chatbot closed.")
            sys.exit(0)

##########################################
#         AudioVisualizer Class          #
##########################################
class AudioVisualizer:
    def __init__(self, audio_file_monitor_path=None, mic_mode=False):
        # Set default path
        global CURRENT_PERSONALITY
        if audio_file_monitor_path is None:
            audio_file_monitor_path = AUDIO_FILE_PATH

        # Path to the audio file used for visualization
        self.audio_file_monitor_path = str(audio_file_monitor_path)
        self.mic_mode = mic_mode
        self.running = True # Control loop for background processing
        self.audio_data = np.zeros(CHUNK_SIZE, dtype=np.float32) # Buffer for audio visualization
        self.is_speaking = False
        self.last_file_mod_time = 0  # Variable for last modified time of monitored audio file 
        self.lock = threading.Lock() # Ensures thread-safe updates to shared data
        self.audio_segment = None
        self.raw_samples = None  # Holds decoded raw audio data
        self.sample_rate = 24000 # Audio sample rate
        self.channels = 1
        self.num_samples = 0 # Total samples in audio
        self.chunk_size = CHUNK_SIZE
        self.format = pyaudio.paInt16 # Format used for microphone input
        self.channels = 1
        self.rate = 24000
        self.current_personality = CURRENT_PERSONALITY

        # Default GUI appearance settings
        self.bg_color = "green"
        self.robot_outline_color = "#C0C0C0"
        self.robot_edge_color = "black"
        self.eye_color = "black"
        self.mouth_color = "#00FFFF"
        self.inner_mouth_color = "#ad4747"

        # Used later for drawing elements
        self.mouth = None
        self.eyebrows = []
        self.force_neutral_mouth = False

        # Start GUI and visualizer logic
        self.setup_gui()
        self.init_visualizer_logic()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Baxter Chatbot Visualizer")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.configure(bg=self.bg_color)

        # UI layout with two frames: one for the visualizer, one for the log
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

        # Set up matplotlib canvas for drawing Baxter's face
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.3, 1.3)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Embed the matplotlib plot into the Tkinter UI
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Chat log UI in bottom frame
        self.log_text = tk.Text(bottom_frame, height=12, wrap='word', bg='black', fg='white')
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(bottom_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = scrollbar.set

        # Redirect Python logs to the GUI log window
        text_handler = TextWidgetHandler(self.log_text)
        text_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        text_handler.setFormatter(formatter)
        logging.getLogger().addHandler(text_handler)

        # Add status bar above terminal
        self.status_var = tk.StringVar()
        self.status_var.set("BAXTER CHAT LOGS")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.CENTER)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.draw_robot_elements()
        self.mouth_visual, = self.ax.plot([], [], color='#00FFFF', linewidth=3, zorder=5)

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def draw_robot_elements(self):
        main_head_width = 1.4
        main_head_height = 1.1
        head_x = -main_head_width / 2.0
        head_y = -main_head_height / 2.0

        self.head = FancyBboxPatch(xy=(head_x, head_y), width=main_head_width, height=main_head_height,
                                   boxstyle="round,pad=0.1,rounding_size=0.5",
                                   facecolor="#FFFFFF", edgecolor=self.robot_edge_color,
                                   linewidth=5, zorder=1)
        self.ax.add_patch(self.head)

        ear_radius = 0.22
        ear_offset = 0.1
        left_ear_center = (head_x - ear_offset, 0.0)
        right_ear_center = (head_x + main_head_width + ear_offset, 0.0)

        self.left_ear = Wedge(center=left_ear_center, r=ear_radius, theta1=90, theta2=270,
                              facecolor="#D3D3D3", edgecolor=self.robot_edge_color, linewidth=5, zorder=1)
        self.right_ear = Wedge(center=right_ear_center, r=ear_radius, theta1=-90, theta2=90,
                               facecolor="#D3D3D3", edgecolor=self.robot_edge_color, linewidth=5, zorder=1)
        self.ax.add_patch(self.left_ear)
        self.ax.add_patch(self.right_ear)

        antenna_bottom_y = 0.65
        antenna_top_y = antenna_bottom_y + 0.2
        self.antenna_line = self.ax.plot([0, 0], [antenna_bottom_y, antenna_top_y],
                                         color=self.robot_edge_color, linewidth=5, zorder=2)[0]
        self.antenna_tip = Circle((0, antenna_top_y), 0.05, fc="#FFFFFF", edgecolor="black", linewidth=5, zorder=3)
        self.ax.add_patch(self.antenna_tip)

        face_width = 1.0
        face_height = 0.75
        face_x = -face_width / 2.0
        face_y = -face_height / 2.0
        self.face = FancyBboxPatch(xy=(face_x, face_y), width=face_width, height=face_height,
                                   boxstyle="round,pad=0.05,rounding_size=0.1",
                                   facecolor="#000000", edgecolor=self.robot_edge_color,
                                   linewidth=5, zorder=2)
        self.ax.add_patch(self.face)

        eye_radius = 0.08
        eye_offset_x = 0.25
        eye_y = 0.10

        self.left_eye = Circle((-eye_offset_x, eye_y), radius=eye_radius, facecolor="#00FFFF", edgecolor=None, zorder=3)
        self.right_eye = Circle((eye_offset_x, eye_y), radius=eye_radius, facecolor="#00FFFF", edgecolor=None, zorder=3)
        self.ax.add_patch(self.left_eye)
        self.ax.add_patch(self.right_eye)

        self.add_mouth_and_eyebrows()

    def add_mouth_and_eyebrows(self):
        self.mouth = None
        self.eyebrows = []
        if self.current_personality in ["friendly", "funny"]:
            return
        if self.current_personality == "evil":
            left_horn_img = mpimg.imread("left_horn.png")
            left_horn_offset = OffsetImage(left_horn_img, zoom=0.2)
            ab_left_horn = AnnotationBbox(left_horn_offset, xy=(-0.45, 0.65), xycoords='data', frameon=False)
            self.ax.add_artist(ab_left_horn)
            right_horn_img = mpimg.imread("right_horn.png")
            right_horn_offset = OffsetImage(right_horn_img, zoom=0.2)
            ab_right_horn = AnnotationBbox(right_horn_offset, xy=(0.45, 0.65), xycoords='data', frameon=False)
            self.ax.add_artist(ab_right_horn)
            left_eyebrow = self.ax.plot([-0.4, -0.2], [0.35, 0.3], color="#00FFFF", lw=3, zorder=10)[0]
            right_eyebrow = self.ax.plot([0.2, 0.4], [0.3, 0.35], color="#00FFFF", lw=3, zorder=10)[0]
            self.eyebrows.extend([left_eyebrow, right_eyebrow])
        elif self.current_personality == "professional":
            pass

    def init_visualizer_logic(self):
        # Create an animation loop that updates the mouth every ~40ms
        self.ani = FuncAnimation(self.fig, self.update_animation, interval=40, blit=False, cache_frame_data=False)

        # Start audio monitoring mode
        if self.mic_mode:
            self.status_var.set("Status: Listening to microphone")
            self.audio_thread = threading.Thread(target=self.process_microphone, daemon=True)
        else:
            self.status_var.set(f"BAXTER CHAT LOGS")
            self.audio_thread = threading.Thread(target=self.monitor_audio_file, daemon=True)
        self.audio_thread.start()
        logging.debug("Visualizer logic started automatically.")

    def update_visual_style(self):
        global CURRENT_PERSONALITY
        if CURRENT_PERSONALITY != self.current_personality:
            logging.debug(f"Personality changed from {self.current_personality} to {CURRENT_PERSONALITY}")
            self.current_personality = CURRENT_PERSONALITY
            self.ax.clear()
            self.ax.set_xlim(-1.2, 1.2)
            self.ax.set_ylim(-1.3, 1.3)
            self.ax.set_aspect('equal')
            self.ax.axis('off')

            if self.current_personality == "friendly":
                new_bg = "green"
                new_face_color = "#C0C0C0"
            elif self.current_personality == "evil":
                new_bg = "#7E191B"
                new_face_color = "#C0C0C0"
            elif self.current_personality == "professional":
                new_bg = "blue"
                new_face_color = "#C0C0C0"
            elif self.current_personality == "funny":
                new_bg = "yellow"
                new_face_color = "#C0C0C0"
            else:
                new_bg = self.bg_color
                new_face_color = self.robot_outline_color

            self.root.configure(bg=new_bg)
            self.fig.patch.set_facecolor(new_bg)
            self.ax.set_facecolor(new_bg)
            self.robot_outline_color = new_face_color
            self.draw_robot_elements()
            self.mouth_visual, = self.ax.plot([], [], color='#00FFFF', linewidth=3, zorder=5)
            self.fig.canvas.draw_idle()
    
    def set_neutral_mouth(self):

        # Forces the mouth into a neutral expression based on the personality.
        # Used when playback stops or no voice is active.
        self.force_neutral_mouth = True
        n = CHUNK_SIZE
        theta = np.linspace(np.pi, 0, n)
        r = 0.3
        center_x, center_y = 0, -0.2

        if self.current_personality in ["friendly", "funny"]:
            vertical_scale = 0.5
            R = r
            x = center_x + R * np.cos(theta)
            y = center_y - (vertical_scale * R) * np.sin(theta)

        elif self.current_personality == "evil":
            vertical_scale = 0.3
            R = r
            x = center_x + R * np.cos(theta)
            y = center_y - (vertical_scale * R) * np.sin(theta)

        elif self.current_personality == "professional":
            x = np.linspace(-0.25, 0.25, n)
            y = np.full(n, -0.2)

        else:
            vertical_scale = 0.5
            R = r
            x = center_x + R * np.cos(theta)
            y = center_y - (vertical_scale * R) * np.sin(theta)

        self.mouth_visual.set_data(x, y)

    def update_animation(self, frame):

        # Updates the shape of the mouth based on real-time audio volume.
        # Called repeatedly by the matplotlib animation loop.
        global STREAMED_AUDIO_DATA
        # If not using mic mode and streamed data is available, update audio_data from it ( fallback )
        if not self.mic_mode and STREAMED_AUDIO_DATA is not None:
            with self.lock:
                self.audio_data = STREAMED_AUDIO_DATA.copy()
        else:
            # Otherwise, audio_data is updated by process_microphone or monitor_audio_file.
            pass

         # Update face style if personality changed
        self.update_visual_style()
        
        # Lock for thread-safe audio data usage
        with self.lock:
            chunk = self.audio_data.copy()
        
        # Log latency from speech start to first visual movement
        if hasattr(self, "tts_start_time") and not getattr(self, "visualizer_sync_logged", False):
            if np.mean(np.abs(self.audio_data)) > 0.01:  # Voice energy is nonzero
                sync_latency = time.time() - self.tts_start_time
                performance_logger.info(f"[Visualizer] Sync latency after TTS start: {sync_latency:.2f} seconds")
                self.visualizer_sync_logged = True
  
        # If in neutral mode, don’t animate
        if self.force_neutral_mouth:
            return

        # Determine loudness of current audio chunk
        volume = np.mean(np.abs(chunk))
        n = len(chunk)
        volume = np.mean(np.abs(chunk))
        mod_factor = np.clip((volume - 0.005) / (0.05 - 0.005), 0, 1)

        # Animate mouth shape based on volume and personality
        if self.current_personality in ["friendly", "funny"]:
            theta = np.linspace(np.pi, 0, n)
            r = 0.3
            center_x, center_y = 0, -0.2
            scale_factor = 0.19
            R = r + (chunk * scale_factor * mod_factor)
            vertical_scale = 0.5
            x = center_x + R * np.cos(theta)
            y = center_y - (vertical_scale * R) * np.sin(theta)
            self.mouth_visual.set_data(x, y)

        elif self.current_personality == "evil":
            theta = np.linspace(np.pi, 0, n)
            r = 0.3
            center_x, center_y = 0, -0.2
            scale_factor = 0.19
            R = r + (chunk * scale_factor * mod_factor)
            vertical_scale = 0.3
            x = center_x + R * np.cos(theta)
            y = center_y - (vertical_scale * R) * np.sin(theta)
            self.mouth_visual.set_data(x, y)

        elif self.current_personality == "professional":
            x = np.linspace(-0.25, 0.25, n)
            base = -0.2
            modulation = chunk * 0.19 * mod_factor
            y = np.full(n, base) - modulation
            self.mouth_visual.set_data(x, y)

        else:
            theta = np.linspace(np.pi, 0, n)
            r = 0.3
            center_x, center_y = 0, -0.2
            scale_factor = 0.19
            R = r + (chunk * scale_factor * mod_factor)
            vertical_scale = 0.5
            x = center_x + R * np.cos(theta)
            y = center_y - (vertical_scale * R) * np.sin(theta)
            self.mouth_visual.set_data(x, y)

        # Simple blink effect
        if frame % 100 == 0 and np.random.random() < 0.3:
            self.left_eye.set_radius(0.05)
            self.right_eye.set_radius(0.05)
        else:
            self.left_eye.set_radius(0.1)
            self.right_eye.set_radius(0.1)

        return

    def on_closing(self):

        # Stops threads and closes the UI safely when the window is closed.
        self.running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        self.root.destroy()
        logging.debug("Application closed")

    def process_microphone(self):

        # Continuously reads from the microphone in real-time and updates the
        # `self.audio_data` buffer for mouth animation.
        try:
            self.p = pyaudio.PyAudio()
            self.chunk_size = CHUNK_SIZE  
            self.stream = self.p.open(format=self.format, channels=self.channels,
                                      rate=self.rate, input=True, frames_per_buffer=self.chunk_size)
            while self.running:
                try:
                    # Read a chunk of raw mic audio (2 bytes per sample)
                    raw_data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                    # Convert raw bytes to integers (signed 16-bit)
                    count = len(raw_data) // 2
                    fmt = f"{count}h"
                    data = struct.unpack(fmt, raw_data)

                    # Normalize audio to float32 [-1, 1] for smoother animation
                    float_chunk = np.array(data).astype(np.float32) / 32768.0

                    # Safely update shared buffer used by animation
                    with self.lock:
                        self.audio_data = float_chunk
                        self.is_speaking = (np.mean(np.abs(self.audio_data)) > 0.01)

                except Exception as e:
                    logging.error(f"Error reading microphone: {e}")
                    time.sleep(0.1)

        except Exception as e:
            logging.error(f"Error in microphone processing: {e}")
        finally:
            # Clean up resources when the loop ends or crashes
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, 'p'):
                self.p.terminate()

    def monitor_audio_file(self):

        # Monitors a static MP3 file and pulls waveform data based on current playback time.
        # This allows real-time animation of the mouth.
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            logging.error(f"Failed to initialize PyAudio: {e}")
            return

        while self.running:
            try:
                # Check if the file has been updated
                if os.path.exists(self.audio_file_monitor_path):
                    mod_time = os.path.getmtime(self.audio_file_monitor_path)
                    if mod_time > self.last_file_mod_time:
                        self.last_file_mod_time = mod_time

                        # Decode audio into raw samples
                        self.audio_segment = AudioSegment.from_mp3(self.audio_file_monitor_path)
                        self.raw_samples = np.array(self.audio_segment.get_array_of_samples(), dtype=np.int16)
                        self.sample_rate = self.audio_segment.frame_rate
                        self.channels = self.audio_segment.channels
                        self.num_samples = len(self.raw_samples)
                        logging.debug(f"Decoded MP3: {self.num_samples} samples, {self.channels} channel(s), {self.sample_rate} Hz")

                    # While pygame is actively playing, update the visualizer based on playback position
                    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                        pos_ms = pygame.mixer.music.get_pos()
                        if pos_ms < 0:
                            pos_ms = 0
                        pos_samples = int((pos_ms / 1000.0) * self.sample_rate) * self.channels
                        chunk_size = CHUNK_SIZE
                        end_samples = pos_samples + chunk_size
                        if (self.raw_samples is not None) and (self.num_samples > 0):
                            if pos_samples < self.num_samples:
                                if end_samples > self.num_samples:
                                    end_samples = self.num_samples
                                chunk = self.raw_samples[pos_samples:end_samples]
                                float_chunk = chunk.astype(np.float32) / 32768.0
                                if self.channels == 2:
                                    float_chunk = float_chunk.reshape(-1, 2)
                                    float_chunk = float_chunk.mean(axis=1)
                                with self.lock:
                                    if len(float_chunk) < CHUNK_SIZE:
                                        padded = np.zeros(CHUNK_SIZE, dtype=np.float32)
                                        padded[:len(float_chunk)] = float_chunk
                                        self.audio_data = padded
                                    else:
                                        self.audio_data = float_chunk[:CHUNK_SIZE]
                                self.is_speaking = True
                            else:
                                # Playback position is beyond audio buffer
                                self.is_speaking = False
                                with self.lock:
                                    self.audio_data = np.zeros(CHUNK_SIZE, dtype=np.float32)
                        else:
                            self.is_speaking = False
                            with self.lock:
                                self.audio_data = np.zeros(CHUNK_SIZE, dtype=np.float32)
                    else:
                        self.is_speaking = False
                        with self.lock:
                            self.audio_data = np.zeros(CHUNK_SIZE, dtype=np.float32)
                else:
                    self.is_speaking = False
                    with self.lock:
                        self.audio_data = np.zeros(CHUNK_SIZE, dtype=np.float32)
            except Exception as e:
                logging.error(f"Error in file monitoring: {e}")
            time.sleep(0.05) # Refresh rate

        # Clean shutdown
        if hasattr(self, 'p'):
            self.p.terminate()

    def run(self):
        self.root.mainloop()

##########################################
#             Terminal GUI               #
##########################################
class TextWidgetHandler(logging.Handler):
    ALLOWED_PHRASES = [
        "Listening for user input",
        "User said:",
        "Response:",
        "Could not understand the audio"
    ]
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    def emit(self, record):
        msg = record.getMessage()
        if any(msg.startswith(phrase) for phrase in self.ALLOWED_PHRASES):
            formatted = self.format(record)
            self.text_widget.after(0, self._append_message, formatted)
    def _append_message(self, msg):
        self.text_widget.insert(tk.END, msg + "\n")
        self.text_widget.see(tk.END)

##########################################
#             Main Section               #
##########################################
def run_ros():

    # ROS Node startup
    rclpy.init()
    node = GPTChatBotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logging.info("ROS node interrupted by keyboard.")
    finally:
        node.shutdown_node()

def main():
    parser = argparse.ArgumentParser(description="Combined Baxter Chatbot and Visualizer")
    parser.add_argument("--file", "-f", help="Path to monitor for audio files for visualizer")
    parser.add_argument("--mic", "-m", action="store_true", help="Use microphone directly for visualizer")
    args = parser.parse_args()

    if args.file:
        audio_file = Path(args.file)
    else:
        audio_file = AUDIO_FILE_PATH

    ros_thread = threading.Thread(target=run_ros, daemon=True)
    ros_thread.start()
    logging.debug("ROS thread started.")

    visualizer = AudioVisualizer(audio_file_monitor_path=audio_file, mic_mode=args.mic)
    import __main__
    __main__.visualizer = visualizer
    visualizer.run()

if __name__ == '__main__':
    main()
