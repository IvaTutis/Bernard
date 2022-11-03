from __future__ import absolute_import, division, print_function, unicode_literals

#importiraj biblioteke
import sys
import os
import re
import numpy as np
from time import time
import tensorflow as tf
import tensorflow_datasets as tfds
tf.keras.utils.set_random_seed(1234)

################################ FUNKCIJE ZA PREPROCESIRANJE I UČITAVANJE PODATAKA ###############################################

#funkcija koja preprocesira rečenicu
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # stvara razmak između riječi i interpunkcije iza nje, npr. "he is smart." => "he is smart ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # produljenje skraćenica (kontrakcija) iz engleskog jezika, npr. "i'm" => "i am"
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # mijenjanje svih znakova koji nisu u skupu (a-z, A-Z, ".", "?", "!", ",") sa znakom razmaka.
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

#funkcija koja učitava skup podataka iz datoteka skinutih sa interneta
def load_conversations():
    # rječnik {key, value} = {id filmske replike, tekst filmske replike}  (op. prev. filmska replika = eng.movie line)
    id2line = {}
    with open(PATH_TO_MOVIE_LINES, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(PATH_TO_MOVIE_CONVERSATIONS, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        # dohvaća razgovor u listi id-a filmskih replika
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs

################################ KLASE I FUNKCIJE VEZANE UZ PAŽNJU (eng. attention)####################################################

#klasa koja predstavlja mehanizam pažnje sa više glava
#sastoji se od četiri dijela:
class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        assert d_model % num_heads == 0
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
            }
        )
        return config

    def split_heads(self, inputs, batch_size):
        inputs = tf.keras.layers.Lambda(
            lambda inputs: tf.reshape(
                inputs, shape=(batch_size, -1, self.num_heads, self.depth)
            )
        )(inputs)
        return tf.keras.layers.Lambda(
            lambda inputs: tf.transpose(inputs, perm=[0, 2, 1, 3])
        )(inputs)

    def call(self, inputs):
        query, key, value, mask = (
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["mask"],
        )
        batch_size = tf.shape(query)[0]

        # Linearni slojevi
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Razdvoji glave
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Skalirani točkasti produkt pažnje
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        )(scaled_attention)

        # Konkatencija glava (heads)
        concat_attention = tf.keras.layers.Lambda(
            lambda scaled_attention: tf.reshape(
                scaled_attention, (batch_size, -1, self.d_model)
            )
        )(scaled_attention)

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights."""
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

######################################### Klase i funkcije vezane uz transformer #################################################

#klasa pozicijsko kodiranje modelu pruža informacije o relativnom položaju riječi u rečenici
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update(
            {
                "position": self.position,
                "d_model": self.d_model,
            }
        )
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )
        # primjeni sinus da na parne indekse u listi
        sines = tf.math.sin(angle_rads[:, 0::2])
        # primjeni kosinus na neparne indekse u listi
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

#pomoćna funkcija za stvaranje maski za maskiranje podstavljenih tokena
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

#pomoćna funkcija za stvaranje maski za maskiranje podstavljenih tokena
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

####################################### FUNKCIJE VEZANE UZ PREDVIĐANJE ODGOVORA MODELA NA REČENICU ########################

#funkcija za predikciju odgovora modela na rečenicu
def predict(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # odaberi zadnju riječ iz dimenzije seq_len
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # vrati rezultat ako je predicted_id jednak end_token - u
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # konkateniraj predicted_id na output koji se predaje dekoderu
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    prediction = tf.squeeze(output, axis=0)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    return predicted_sentence

#######################################FUNKCIJE VEZANE ZA DETEKCIJU KRAJA RAZGOVORA####################################

#funkcija koja ispituje je li kraj razgovora detektiran (to jest, da li korisnik žei otići) nakon unosa rečenice
def user_may_want_to_leave(sentence):
    global CONVERSATION_END_WORD_LIST

    for word in CONVERSATION_END_WORD_LIST:
        #print("the end word is ["+word+"], while the user input is ["+sentence+"]")
        if (is_word_in_sentence(word,sentence)):
            return True
    return False

#funkcija koja ispituje je li korisnik nezainteresiran ili zainteresiran za nastavak razgovora nakon unosa rečenice
def user_is_uninterested(sentence):
    global user_interest_level
    global max_user_interest_level

    #ukoliko je korisnik nezainteresiran, te varijable user_interest_level = 0, vrati da je nezainteresiran
    if (user_interest_level==0):
        #sa obzirom da će ovo ili biti kraj razgovora ili potvrda nastavka, resetiraj user_interest_level na baseline razinu (maksimum)
        user_interest_level=BASELINE_USER_INTEREST_LEVEL
        return True;

    return False;

#funkcija koja rekalkulira razinu interesa korisnika nakon unosa rečenice
def recalculate_users_interest_level(sentence):
    global user_interest_level

    word_list = sentence.split()
    number_of_words = len(word_list)

    #svaki put kada korisnik odgovori sa 2 riječi ili manje, oduzima se bod, osim ako je razina interesa već 0
    #svaki put kada korisnik odgovori sa više od 2 riječi, dodaje se bod
    if (number_of_words<3):
        if(user_interest_level>0):
            user_interest_level-=1
    else:
         user_interest_level+=1

#funkcija koja ispituje je li potrebno izazvati (eng. prompt) upit za kraj razgovora
def prompt_end_of_conversation(sentence):
    if (user_is_uninterested(sentence)==True or user_may_want_to_leave(sentence)==True):
        return True
    return False

#funkcija koja ispituje da li se riječ [word] nalazi u rečenici [sentence]
def is_word_in_sentence(word, sentence):
    sentence=preprocess_sentence(sentence)
    word_list = sentence.split()

    if word in word_list:
        return True

    return False

#################################################CHATBOT BEGINS HERE####################################################

#importiraj parametre
from parametri import *

#########################učitaj model########################
model = tf.keras.models.load_model(
    PATH_TO_MODEL,
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)

print("Loaded model...")

################omogući tokeniziranje##########################

#učitaj pitanja i odgovore iz kojih ćemo sagraditi vokabular
questions, answers = load_conversations()

#Sagradi tokenizer
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13
)
#Definiraj početni i završni token tako da indiciraju početak i kraj rećenice
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

#Veličina vokabulara je sad veličina vokabulara + {početni token, završni token}
VOCAB_SIZE = tokenizer.vocab_size + 2

print("Made tokenizer ...")

################ postavi baseline razinu interesa korisnika ##########################

# definiraj i inicijaliziraj globalnu varijablu koja označava razinu interesa korisnika
user_interest_level=BASELINE_USER_INTEREST_LEVEL

print("User interest level set to baseline ...")

################ kod za razgovor u konzoli ( while(True) petlja ) ##########################
print("Coversation starting ...")

while(True):
    #korisnik unosi string u konzolu
    print(USER_NAME)
    sentence = input()

    #rekalkuliraj interes korisnika nakon njihovog unosa
    recalculate_users_interest_level(sentence)
    #print("Current user's interest level is "+ str(user_interest_level))

    #####chatbot odgovara na unos korisnika#####

    #provjeravamo je li došlo do kraja razgovora
    end_conversation_flag=prompt_end_of_conversation(sentence)
    #print("The end-conversation-prompt was "+ str(end_conversation_flag))

    if (end_conversation_flag):
        while(True):
            #chatbot model answers
            print(CHATBOT_NAME)
            print("Do you want to end this conversation? Answer 'yes' or 'no'.")

            #i read your input
            print(USER_NAME)
            prompt_answer=input()

            if (is_word_in_sentence("yes", prompt_answer)):
                print(CHATBOT_NAME)
                print("Ok. Goodbye.")
                break
            elif (is_word_in_sentence("no", prompt_answer)):
                print(CHATBOT_NAME)
                print("Okay. Glad I can keep talking to you.")
                end_conversation_flag=False
                break
            else:
                print(CHATBOT_NAME)
                print("I didn't catch that.")
        #u slučaju da je korisnik potvrdio da želi završiti razgovor
        if(end_conversation_flag):
            break
        else:
            continue

    #ukoliko prompt za kraj razgovora nije okinut, odgovor chatbota se generira uz pomoć natreniranog modela
    print(CHATBOT_NAME)
    print(predict(sentence))