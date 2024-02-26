import speech_recognition as sr
import pyttsx3 
 
r = sr.Recognizer() 
 
def SpeakText(command):
     
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
     
     
SpeakText("Why did you leave me to get milk on that dark stormy night")