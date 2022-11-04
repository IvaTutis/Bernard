import tensorflow as tf
import os

#path do zip datoteke skupa podataka
PATH_TO_ZIP = tf.keras.utils.get_file(
    "cornell_movie_dialogs.zip",
    origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
    extract=True,
)

#path do direktorija skupa podataka
PATH_TO_DATASET = os.path.join(
    os.path.dirname(PATH_TO_ZIP), "cornell movie-dialogs corpus"
)

#path - ovi do datoteka koji sadrže filmske replike i filmske razgovore
PATH_TO_MOVIE_LINES = os.path.join(PATH_TO_DATASET, "movie_lines.txt")
PATH_TO_MOVIE_CONVERSATIONS = os.path.join(PATH_TO_DATASET, "movie_conversations.txt")

# Maksimalna duljina rečenice
MAX_LENGTH = 40

# Maksimalni broj sample-ova za preprocesirati
MAX_SAMPLES = 250000

#Početna (eng. baseline) razina interesa korisnika
BASELINE_USER_INTEREST_LEVEL=5

#Lista izraza koji se koriste kao indikacija za kraj razgovora
CONVERSATION_END_WORD_LIST = ["bye", "goodbye", "bye for now", "see you", "see ya", "be seeing you", "see you soon", "i'm off", "i am off","cheerio", "catch you later", "good night", "farewell", "adieu", "adios", "Godspeed"]

#Ime korisnika koje se prikazuje tokom razgovora
USER_NAME="You:"

#Ime chatbota koje se prikazuje tokom razgovora
CHATBOT_NAME= "Bernard:"

#Path do pretreniranog modela
PATH_TO_MODEL="NatreniraniModeli/model3.h5"