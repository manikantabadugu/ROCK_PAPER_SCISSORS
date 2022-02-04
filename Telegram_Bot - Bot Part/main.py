
import os
import cv2
import telebot
import random
import json
import joblib
import numpy as np
import pandas as pd
#from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append(r"C:\Users\User\Documents\GitHub\practice folders\Buildweek2.1\Telegram Bot Project\Telegram Bot - Machine Learning\handgestures")
from handgestures.transform_image import transform_single_image

with open(r"C:\Users\User\Documents\GitHub\practice folders\Buildweek2.1\Telegram Bot Project\Telegram_Bot - Bot Part\config.json") as f:
 token = json.load(f)

bot = telebot.TeleBot(token["telegramToken"])
x = bot.get_me()
print(x)


choices = ['rock', 'paper', 'scissors']
computer_choice = random.choice(choices)


@bot.message_handler(commands=['play'])
def start(message):
   bot.send_message(message.chat.id, '''write Rock or paper or scissors or Upload an Image of your hand showing one of the gestures:
   -rock 
   -paper
   -scissors ''')

def user_input(message):
  request = message.text.lower()
  if request not in ['rock', 'paper', 'scissors']:
    return False
  else:
    return True

@bot.message_handler(content_types=['photo'])
def photo(message):
    
    fileID = message.photo[-1].file_id
    path = "image.jpg"

    file_info = bot.get_file(fileID)

    downloaded_file = bot.download_file(file_info.file_path)


    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    image = cv2.imread(r'C:\Users\User\Documents\GitHub\practice folders\Buildweek2.1\Telegram Bot Project\ROCK_PAPER_SCISSORS\Telegram Bot - Machine Learning\image.jpg')
    #Transform the Image into a Skeleton
    img = transform_single_image(image)
    
    #These conditions are here to ensure that image is properly transformed
    if len(img.shape)== 3:
      bot.send_message(message.chat.id, "Image is not clear, please upload another picture, not too close to camera")
      return start(message) 
    else:
      pass


    if img is None: 
      print(img)
      bot.send_message(message.chat.id, "Image is not clear, please upload another picture, not too close to camera")
      return start(message) 
    else:
      pass
      


    print(img.shape)


    #Transform the Skeleton into a flattened array
    img_arr = np.array(img, dtype = int)
    img_arr = img_arr.flatten()
    class_data_trail= pd.DataFrame(img_arr)
    class_data_trail=class_data_trail.transpose()

    #import the model from best_model_2.sav
    model_choice = 'best_model_2.sav'
    loaded_model = joblib.load(model_choice)

    #Predict the Input Image into a Class (Rock, Paper, or Scissors)
    rps_class = loaded_model.predict(class_data_trail)

    if int(rps_class[0]) == 0:
      bot.send_message(message.chat.id, "You Give ROCK!")
      player_choice = 'rock'
    elif int(rps_class[0]) == 1:
      bot.send_message(message.chat.id, "You Give PAPER!")
      player_choice = 'paper'  
    else:
      bot.send_message(message.chat.id, "You Give SCISSORS!")
      player_choice = 'scissors'

    computer_choice = random.choice(choices)

    bot.send_message(message.chat.id,"PC picked: %s" % computer_choice)  

    if player_choice == computer_choice:
      bot.send_message(message.chat.id, "It's a Tie")      
    elif player_choice == 'rock' and computer_choice == 'scissors':
      bot.send_message(message.chat.id, "Player wins!")
    elif player_choice == 'scissors' and computer_choice == 'paper':
      bot.send_message(message.chat.id, "Player wins!")
    elif player_choice == 'paper' and computer_choice == 'rock':
      bot.send_message(message.chat.id, "Player wins!")
    else:
      bot.send_message(message.chat.id, "PC wins!")  







@bot.message_handler(func=user_input)
def send_output(message):
  player_choice = message.text.lower()

  computer_choice = random.choice(choices)

  if player_choice == computer_choice:
    bot.send_message(message.chat.id, "It's a Tie")      
  elif player_choice == 'rock' and computer_choice == 'scissors':
      bot.send_message(message.chat.id, "Player wins!")
  elif player_choice == 'scissors' and computer_choice == 'paper':
      bot.send_message(message.chat.id, "Player wins!")
  elif player_choice == 'paper' and computer_choice == 'rock':
      bot.send_message(message.chat.id, "Player wins!")
  else:
      bot.send_message(message.chat.id,"PC picked: %s" % computer_choice)  
      bot.send_message(message.chat.id, "PC wins!")
      
    




bot.polling()