import speech_recognition as sr
import pyttsx3 
 
r = sr.Recognizer() 
 
def SpeakText(command):
     
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
     
     
while(1):    
    try:
         
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration = 1)
             
            audio2 = r.listen(source2)
             
            text = r.recognize_google(audio2)
            text = MyText.lower()
 
            print(text)
            SpeakText(MyText)
             
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
         
    except sr.UnknownValueError:
        print("unknown error occurred")