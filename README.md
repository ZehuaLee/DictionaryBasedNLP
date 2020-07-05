# DictionaryBasedNLP
the project is to *1. get text from pdf* and *2. then classify them to different types*. 

##pdf2jpg
Turn the raw data of pdf to jpg with a fixed size/resolutions.
## lib/Extractor
Firstly scrapping the text from image by Google OCR API, and then get the coordinates as well as text.

## lib/analyzer
Get the text from Extractor and then create a dictionary, by contiously compare the distance between the input text and labeled text in dictionary, judge the classification of the unknow text.
